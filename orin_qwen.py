#!/usr/bin/env python3
"""
Орин — полностью локальный голосовой ассистент на Rockchip RK3588
Технологии: Sherpa-onnx (ASR), MMS-TTS (TTS), Qwen-1.7B (LLM), rknn, asyncio, sounddevice
Не использует облака. Не требует интернета. Не слышит себя.
"""

import os
import time
import numpy as np
import sounddevice as sd
import asyncio
import queue
import json
from typing import Optional
import requests
from requests import Session
from normalizer import StreamTextProcessor

# =========================
# 📦 ИМПОРТ ВАШИХ МОДУЛЕЙ — УБЕДИТЕСЬ, ЧТО ПУТИ ВЕРНЫ
# =========================
from mms_tts import TTSVocaliser, play_audio_resample
from vosk_recognizer_async import listen_and_recognize_phrase


# =========================
# 🛠️ КОНФИГУРАЦИЯ — ЗАМЕНИТЕ НА СВОИ ПУТИ
# =========================
MODEL_DIR = "/home/pi/Repo/orin/models"
SERVER_URL = 'http://192.168.1.17:8080/rkllm_chat'
LLM_MODEL = "Qwen3-0.6B-rk3588-w8a8.rkllm"
MAX_LENGTH = 200
vocab = {}  # Загрузите свой словарь из mms_tts.py, если требуется

# =========================
# 📦 ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ — ОДИН РАЗ, НА ВСЁ ВРЕМЯ
# =========================
text_queue = queue.Queue()            # синхронная очередь для текста (от LLM → TTS)
audio_buffer = asyncio.Queue()        # асинхронная очередь для готового аудио (TTS → player)
expected_audio_count = 0              # 🔑 КЛЮЧЕВОЙ: сколько аудиофрагментов ожидается
audio_count_lock = asyncio.Lock()     # блокировка для безопасного доступа к счётчику
is_running = True                     # флаг остановки — используется только при выходе

tts_vocaliser = TTSVocaliser()        # один экземпляр на всё время
session = Session()                   # одна сессия на всё время


# =========================
# 🤖 LLM — запрос к локальному серверу Qwen-1.7B
# =========================
def send_chat_request_queued(user_message: str, is_streaming=True):
    global expected_audio_count
    """Отправляет запрос в LLM и кладёт ответ в text_queue по фрагментам"""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'not_required'
    }
    data = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": user_message}],
        "stream": is_streaming,
        "enable_thinking": False,
        "tools": None
    }

    print(f"📩 [LLM] Отправляю запрос: {user_message[:50]}...")
    responses = session.post(SERVER_URL, json=data, headers=headers, stream=is_streaming, verify=False)

    if responses.status_code != 200:
        print(f"❌ [LLM] Ошибка: {responses.text}")
        return

    # 👇 Создаём один процессор на весь ответ
    processor = StreamTextProcessor(max_chunk_size=70)
    buff = ""

    for line in responses.iter_lines():
        if not line:
            continue
        try:
            line_data = json.loads(line.decode('utf-8'))
        except json.JSONDecodeError:
            continue

        if line_data["choices"][-1]["finish_reason"] == "stop":
            # 👇 ДОБАВЛЯЕМ ОСТАТОК В buff — чтобы обработать его в процессоре
            if buff.strip():
                for char in buff:
                    processor.feed(char)
                buff = ""

            # 👇 КЛЮЧЕВОЙ ШАГ: ИЗВЛЕКАЕМ ВСЁ, ЧТО ОСТАЛОСЬ В ПРОЦЕССОРЕ
            fragments = processor.flush()
            for frag in fragments:
                if frag:
                    text_queue.put(frag)
                    print(f"📦 [LLM] Последний фрагмент: '{frag[:40]}...' (#{len(frag)})")
                    expected_audio_count += 1
            break  # ← Выход из цикла

        reply = line_data["choices"][-1]["delta"].get("content", "")
        if reply:
            # 👇 Потоковая обработка каждого символа
            for char in reply:
                fragments = processor.feed(char)
                for frag in fragments:
                    if frag:
                        text_queue.put(frag)
                        print(f"📦 [LLM] Фрагмент: '{frag[:40]}...' (#{len(frag)})")
                        expected_audio_count += 1

            # Добавляем reply в buff — для последней обработки (если finish_reason не пришёл)
            buff += reply

    # 👇 🔴 БЕЗ ЭТОГО — ПОСЛЕДНЕЕ СЛОВО ПРОПАДАЕТ!
    # НО: мы уже вызвали flush() в `finish_reason == "stop"` — так что здесь не нужно
    # Однако, если LLM вернёт ответ без "finish_reason" — это резервная защита:
    if not line_data["choices"][-1]["finish_reason"]:  # редкий случай
         fragments = processor.flush()
         for frag in fragments:
            if frag:
                text_queue.put(frag)
                expected_audio_count += 1


# =========================
# 🚀 АСИНХРОННЫЙ КОНВЕЙЕР — РАБОТАЕТ ВСЕГДА
# =========================
async def audio_player():
    """Проигрывает аудиофрагменты из очереди — НИКОГДА НЕ УМИРАЕТ"""
    print("🔊 [AUDIO PLAYER] Запущен (всегда работает)")
    try:
        while True:
            audio_data = await audio_buffer.get()
            if audio_data is None:
                print("🛑 [AUDIO PLAYER] Получен сигнал остановки — завершаю...")
                break

            print(f"🎧 [PLAYER] Начинаю проигрывание фрагмента длиной {len(audio_data)} сэмплов")
            start_play = time.time()
            await asyncio.to_thread(play_audio_resample, audio_data)
            duration_sec = len(audio_data) / 16000
            end_play = time.time()

            print(f"✅ [PLAYER] Проигрывание завершено (реальное время: {duration_sec:.3f}с)")
            print(f"⏱️ [PLAYER] play_audio() заняла {end_play - start_play:.3f}с (ожидалось ~{duration_sec:.3f}с)")

            audio_buffer.task_done()

            # 👇 Уменьшаем счётчик ожидаемых фрагментов — один проигран
            async with audio_count_lock:
                global expected_audio_count
                expected_audio_count -= 1
                print(f"📊 [PLAYER] Осталось ожидать: {expected_audio_count} фрагментов")
    except Exception as e:
        print(f"❌ [AUDIO PLAYER] Ошибка: {e}")
    finally:
        print("✅ [AUDIO PLAYER] Завершён")


async def audio_synthesizer():
    """Синтезирует текст в аудио — НИКОГДА НЕ УМИРАЕТ"""
    print("🧠 [SYNTHESIZER] Запущен (всегда работает)")
    try:
        while True:
            text = await asyncio.to_thread(text_queue.get)
            if text is None:
                print("🛑 [SYNTHESIZER] Получен сигнал остановки — завершаю...")
                await audio_buffer.put(None)
                break

            print(f"📝 [SYNTHESIZER] Получил текст: '{text}'", flush=True)
            audio_data = tts_vocaliser.synthesize(text)
            print(f"🎵 [SYNTHESIZER] Синтезировал: {len(audio_data)} сэмплов → кладу в буфер", flush=True)
            #await audio_buffer.put(audio_data)
            audio_buffer.put_nowait(audio_data)
            print(f"🎵 [SYNTHESIZER] Положил в буффер, жду текста", flush=True)

            # 👇 Уменьшаем счётчик ожидаемых фрагментов — один синтезирован и положен
            #async with audio_count_lock:
            #    global expected_audio_count
            #    expected_audio_count -= 1
            #    print(f"📊 [SYNTHESIZER] Осталось ожидать: {expected_audio_count} фрагментов")
    except Exception as e:
        print(f"❌ [SYNTHESIZER] Ошибка: {e}")
        await audio_buffer.put(None)
    finally:
        print("✅ [SYNTHESIZER] Завершён")


# =========================
# 📦 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================
async def say_message(msg: str):
    """Кладёт текст в очередь — не озвучивает сразу!"""
    async with audio_count_lock:
        global expected_audio_count
        expected_audio_count += 1
    print(f"📤 [SAY] Отправляю в очередь: '{msg}'")
    text_queue.put(msg)

def clear_text_queue():
    """Очищает очередь текста — убирает мусор перед новым запросом"""
    while not text_queue.empty():
        try:
            text_queue.get_nowait()
        except queue.Empty:
            break
    print("🧹 Очередь text_queue очищена.")


# =========================
# 🚀 ГЛАВНЫЙ ЦИКЛ — ОДИН РАЗ, НА ВСЮ ЖИЗНЬ
# =========================
async def run_agent():
    global is_running

    # 🔧 ИНИЦИАЛИЗАЦИЯ
    print("📢 Инициализация...")
    print("✅ TTS инициализирован")
    print("✅ ASR инициализирован")
    print("✅ LLM сервер запущен")

    # 📣 ПРИВЕТСТВИЕ
    greetings = [
        "Привет, я Орин.",
        "Я — ваш голосовой ассистент.",
        "Я работаю полностью локально."
    ]

    print("📢 Приветствую... Отправляю приветственные сообщения в очередь...")
    for msg in greetings:
        await say_message(msg)

    # 🚀 ЗАПУСК КОНВЕЙЕРА — ОДИН РАЗ, НА ВСЁ ВРЕМЯ
    print("🚀 Запускаю асинхронный конвейер (постоянно работает)...")
    player_task = asyncio.create_task(audio_player())
    synth_task = asyncio.create_task(audio_synthesizer())

    # 👇 ЖДЁМ, ПОКА ПРИВЕТСТВИЯ ПРОИГРАЮТСЯ
    print("⏳ Жду, пока приветственные сообщения проиграются...")
    while True:
        async with audio_count_lock:
            if expected_audio_count == 0:
                break
        await asyncio.sleep(0.1)
    print("✅ Все приветственные сообщения проиграны! Теперь можно слушать.")

    # 👇 ОСНОВНОЙ ЦИКЛ — СЛУШАЕМ, ОТВЕЧАЕМ — НИКОГДА НЕ ЗАВЕРШАЕТСЯ
    try:
        while True:
            print("\n--- Ожидаю речи... ---")
            inp = await listen_and_recognize_phrase(timeout=15.0)

            if not inp.strip():
                print("🔇 Пустой ввод, пропускаем...")
                continue

            # ✅ ОЧИЩАЕМ ОЧЕРЕДЬ — чтобы не было "старых" фрагментов
            clear_text_queue()

            print(f"📩 Отправляем запрос: {inp}")
            await asyncio.to_thread(send_chat_request_queued, inp, True)

            # 👇 ЭТАП 1: ЖДЁМ, ПОКА LLM ЗАКОНЧИЛ ОТПРАВЛЯТЬ ТЕКСТ
            print("⏳ Жду, пока LLM закончит отправлять фрагменты...")
            while not text_queue.empty():
                await asyncio.sleep(0.1)
            print("✅ Все фрагменты текста в очереди.")

            # 👇 ЭТАП 2: ЖДЁМ, ПОКА ВСЕ ФРАГМЕНТЫ БУДУТ ПРОИГРАНЫ
            print("⏳ Жду, пока все фрагменты будут проиграны...")
            while True:
                async with audio_count_lock:
                    if expected_audio_count == 0:
                        break
                print(f"⏳ Осталось {expected_audio_count} фрагментов — жду...")
                await asyncio.sleep(0.2)
            print("✅ Все фрагменты проиграны!")

            # 👇 ЭТАП 3: ЖДЁМ РЕАЛЬНОЕ ВРЕМЯ ПРОИГРЫВАНИЯ — чтобы эхо затухло
            print("⏸️ Жду, чтобы звук реально затух... (1.5с)")
            await asyncio.sleep(1.5)

            print("✅ Ответ проигран. Можно слушать снова.")

            # 👇 ТОЛЬКО ТЕПЕРЬ — МИКРОФОН ВКЛЮЧАЕТСЯ
            print("\n--- Ожидаю речи... ---")

    except KeyboardInterrupt:
        print("\n🛑 Остановка агента...")
        is_running = False
        text_queue.put(None)  # ← Сигнал остановки для синтезатора
        await asyncio.sleep(0.5)  # Дать время на обработку
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        is_running = False
        text_queue.put(None)
        await asyncio.sleep(0.5)


# =========================
# 🏁 ЗАПУСК
# =========================
if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\n👋 Программа остановлена пользователем.")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")