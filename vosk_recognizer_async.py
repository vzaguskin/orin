MODEL_DIR = "models/vosk-asr"

import asyncio
import time
import sherpa_onnx
import sounddevice as sd
import os
import queue  # синхронная очередь для передачи данных из потока в asyncio


# Глобальный кэш распознавателя
_recognizer_cache = None

def create_recognizer():
    global _recognizer_cache
    if _recognizer_cache is not None:
        return _recognizer_cache

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=os.path.join(MODEL_DIR, "tokens.txt"),
        encoder=os.path.join(MODEL_DIR, "vosk-ru-encoder.rknn"),
        decoder=os.path.join(MODEL_DIR, "vosk-ru-decoder.rknn"),
        joiner=os.path.join(MODEL_DIR, "vosk-ru-joiner.rknn"),
        num_threads=4,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=1.0,
        rule2_min_trailing_silence=0.8,
        rule3_min_utterance_length=3.0,
        provider="rknn",
        model_type="zipformer",
        debug=False,
    )
    _recognizer_cache = recognizer
    return recognizer


async def listen_and_recognize_phrase(device=0, sample_rate=16000, timeout=15.0):
    """
    Асинхронная версия распознавания речи.
    Использует синхронную очередь для передачи данных из потока в asyncio.
    """
    sync_queue = queue.Queue()  # ← СИНХРОННАЯ очередь — безопасна между потоками
    stop_event = asyncio.Event()

    def recognition_worker():
        """Работает в отдельном потоке"""
        try:
            recognizer = create_recognizer()
            stream = recognizer.create_stream()

            print("🎙️ Говорите... (ждём конца фразы)")

            # 👇 audio_callback теперь внутри recognition_worker — захватывает stream и sample_rate
            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"⚠️ Audio stream status: {status}")
                samples = indata[:, 0]
                stream.accept_waveform(sample_rate, samples)

            with sd.InputStream(
                device=device,
                channels=1,
                samplerate=sample_rate,
                dtype='float32',
                callback=audio_callback,
                blocksize=4000
            ):
                start_time = time.time()

                while not stop_event.is_set():
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)

                    if recognizer.is_endpoint(stream):
                        result = recognizer.get_result(stream).strip()
                        if result:
                            print(f"✅ Распознано: {result}")
                            sync_queue.put(result)  # ← Кладём в синхронную очередь!
                        recognizer.reset(stream)

                    time.sleep(0.01)

                    if time.time() - start_time > timeout:
                        print("⏰ Таймаут — ничего не распознано.")
                        sync_queue.put("")  # ← Сигнал таймаута
                        break

        except Exception as e:
            print(f"❌ Ошибка в recognition_worker: {e}")
            sync_queue.put("")  # ← В случае ошибки тоже кладём пустой результат
        finally:
            # Убедимся, что очередь не зависнет
            sync_queue.put(None)  # ← Сигнал завершения

    # Запускаем worker в отдельном потоке
    loop = asyncio.get_running_loop()
    worker_task = loop.run_in_executor(None, recognition_worker)

    # Асинхронно читаем из синхронной очереди
    try:
        while True:
            # Ждём результат из синхронной очереди — без блокировки event loop
            result = await asyncio.to_thread(sync_queue.get)  # ← Это НЕ корутина, а синхронный get()

            if result is None:
                # Получили сигнал завершения
                break
            elif result == "":
                # Таймаут или пустой ввод
                print("🔇 Пустой ввод")
                return ""
            else:
                # Успешно распознано
                stop_event.set()  # Останавливаем worker
                await worker_task  # Ждём завершения потока
                return result

    except Exception as e:
        print(f"❌ Ошибка чтения из очереди: {e}")
        stop_event.set()
        await worker_task
        return ""

    # Если вышли из цикла — возвращаем пустое значение
    return ""