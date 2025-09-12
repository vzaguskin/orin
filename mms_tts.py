import numpy as np
from rknn.api import RKNN
import os
from tts_vocab import vocab
import sounddevice as sd
import torch
from torch import nn
import soundfile as sf

MAX_LENGTH = 200
MODEL_DIR = "/home/pi/Repo/rknn_model_zoo/examples/mms_tts/ru"
ENCODER = "mms_tts_eng_encoder_200.rknn"
DECODER = "mms_tts_eng_decoder_200.rknn"

def run_encoder(encoder_model, input_ids_array, attention_mask_array):
    if 'rknn' in str(type(encoder_model)):
        log_duration, input_padding_mask, prior_means, prior_log_variances = encoder_model.inference(inputs=[input_ids_array, attention_mask_array])
    elif 'onnx' in str(type(encoder_model)):
        log_duration, input_padding_mask, prior_means, prior_log_variances = encoder_model.run(None, {"input_ids": input_ids_array, "attention_mask": attention_mask_array})

    return log_duration, input_padding_mask, prior_means, prior_log_variances

def run_decoder(decoder_model, attn, output_padding_mask, prior_means, prior_log_variances):
    if 'rknn' in str(type(decoder_model)):
        waveform  = decoder_model.inference(inputs=[attn, output_padding_mask, prior_means, prior_log_variances])[0]
    elif 'onnx' in str(type(decoder_model)):
        waveform  = decoder_model.run(None, {"attn": attn, "output_padding_mask": output_padding_mask, "prior_means": prior_means, "prior_log_variances": prior_log_variances})[0]

    return waveform

def pad_or_trim(token_id, attention_mask, max_length):
    pad_len = max_length - len(token_id)
    if pad_len <= 0:
        token_id = token_id[:max_length]
        attention_mask = attention_mask[:max_length]

    if pad_len > 0:
        token_id = token_id + [0] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    return token_id, attention_mask

def preprocess_input(text, vocab, max_length):
    text = list(text.lower())
    input_id = []
    for token in text:
        if token not in vocab:
            continue
        input_id.append(0)
        input_id.append(int(vocab[token]))
    input_id.append(0)
    attention_mask = [1] * len(input_id)

    input_id, attention_mask = pad_or_trim(input_id, attention_mask, max_length)

    input_ids_array = np.array(input_id)[None,...]
    attention_mask_array = np.array(attention_mask)[None,...]

    return input_ids_array, attention_mask_array

def middle_process(log_duration, input_padding_mask, max_length):
    log_duration = torch.tensor(log_duration)
    input_padding_mask = torch.tensor(input_padding_mask)

    speaking_rate = 1
    length_scale = 1.0 / speaking_rate
    duration = torch.ceil(torch.exp(log_duration) * input_padding_mask * length_scale)
    predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()

    # Create a padding mask for the output lengths of shape (batch, 1, max_output_length)
    predicted_lengths_max_real = predicted_lengths.max()
    predicted_lengths_max = max_length * 2

    indices = torch.arange(predicted_lengths_max, dtype=predicted_lengths.dtype)
    output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
    output_padding_mask = output_padding_mask.unsqueeze(1).to(input_padding_mask.dtype)

    # Reconstruct an attention tensor of shape (batch, 1, out_length, in_length)
    attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(output_padding_mask, -1)
    batch_size, _, output_length, input_length = attn_mask.shape
    cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
    indices = torch.arange(output_length, dtype=duration.dtype)
    valid_indices = indices.unsqueeze(0) < cum_duration
    valid_indices = valid_indices.to(attn_mask.dtype).view(batch_size, input_length, output_length)
    padded_indices = valid_indices - nn.functional.pad(valid_indices, [0, 0, 1, 0, 0, 0])[:, :-1]
    attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask

    attn = attn.numpy()
    output_padding_mask = output_padding_mask.numpy()
    
    return attn, output_padding_mask, predicted_lengths_max_real

def play_audio(waveform, samplerate=16000):
    """
    Проигрывает аудио из numpy-массива через наушники.
    
    waveform: numpy.ndarray формы (n_samples,) или (n_samples, channels)
    samplerate: частота дискретизации (например, 16000)
    """
    # Нормализуем аудио в диапазон [-1, 1] (если не нормализовано)
    waveform = np.asarray(waveform, dtype=np.float32)
    if np.max(np.abs(waveform)) > 1.0:
        waveform = waveform / np.max(np.abs(waveform))

    print("🔊 Проигрывание аудио...")
    print(waveform.shape, max(waveform), waveform.mean())
    sd.play(waveform, samplerate=samplerate, device=0)
    sd.wait()  # Ждём окончания проигрывания
    print("✅ Проигрывание завершено.")

class TTSVocaliser:
    def __init__(self):
        encoder_path = os.path.join(MODEL_DIR, ENCODER)
        decoder_path = os.path.join(MODEL_DIR, DECODER)
        self.encoder = self.init_rknn_model(encoder_path)
        self.decoder = self.init_rknn_model(decoder_path)

    def init_rknn_model(self, model_path):
        model = RKNN()
        ret = model.load_rknn(model_path)
        ret = model.init_runtime(target="rk3588", device_id=None)
        return model
    
    def vocalise(self, inp: str):
        input_ids_array, attention_mask_array = preprocess_input(inp, vocab, max_length=MAX_LENGTH)

        # Encode
        log_duration, input_padding_mask, prior_means, prior_log_variances = run_encoder(self.encoder, input_ids_array, attention_mask_array)

        # Middle process
        attn, output_padding_mask, predicted_lengths_max_real = middle_process(log_duration, input_padding_mask, MAX_LENGTH)

        # Decode
        waveform = run_decoder(self.decoder, attn, output_padding_mask, prior_means, prior_log_variances)
        #audio_save_path = "test_output.wav"
        #sf.write(file=audio_save_path, data=np.array(waveform[0][:predicted_lengths_max_real * 256]), samplerate=16000)
    
        audio_data=np.array(waveform[0][:predicted_lengths_max_real * 256])
        play_audio(audio_data)


if __name__ == "__main__":
    voc = TTSVocaliser()
    voc.vocalise("Привет, я ассистент Орин, я говорю по русски")



