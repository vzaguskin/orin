import sherpa_onnx
import time
import sounddevice as sd
import numpy as np
import os

MODEL_DIR = "models/vosk-asr"

import sherpa_onnx

def create_recognizer():
    # Создаём распознаватель напрямую через конструктор
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens = os.path.join(MODEL_DIR, "tokens.txt"),
        encoder = os.path.join(MODEL_DIR, "vosk-ru-encoder.rknn"),
        decoder = os.path.join(MODEL_DIR, "vosk-ru-decoder.rknn"),
        joiner = os.path.join(MODEL_DIR, "vosk-ru-joiner.rknn"),
        num_threads=4,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,  # ← Включаем детекцию конца фразы!
        rule1_min_trailing_silence=1.0,  # 1 сек тишины = конец фразы
        rule2_min_trailing_silence=0.8,
        rule3_min_utterance_length=3.0,  # мин. длина фразы 3 сек
        provider="rknn",  # ← важно для Rockchip!
        model_type="zipformer",
        debug=False,
    )
    return recognizer



def listen_and_recognize_phrase(device=0, sample_rate=16000, timeout=15.0):
    recognizer = create_recognizer()
    stream = recognizer.create_stream()

    print("🎙️ Говорите... (ждём конца фразы, таймаут 15 сек)")

    start_time = time.time()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        samples = indata[:, 0]
        stream.accept_waveform(sample_rate, samples)

    with sd.InputStream(
        device=device,
        channels=1,
        samplerate=sample_rate,
        dtype='float32',
        callback=callback,
        blocksize=4000
    ):
        while time.time() - start_time < timeout:
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            if recognizer.is_endpoint(stream):
                result = recognizer.get_result(stream).strip()
                if result:
                    print(f"✅ Распознано: {result}")
                    return result
                recognizer.reset(stream)

            time.sleep(0.01)

    print("⏰ Таймаут — ничего не распознано.")
    return ""