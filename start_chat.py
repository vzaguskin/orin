from pipeline import start_pipeline, text_queue, stop_pipeline, audio_buffer
from vosk_recognizer_async import listen_and_recognize_phrase
from llm_chat import send_chat_request, send_chat_request_queued
from mms_tts import TTSVocaliser
import asyncio
import queue
import os

# --- Глобальные переменные ---
WELCOME_MESSAGE = "Привет, я Орин"
# ← ИНИЦИАЛИЗИРУЕМ ДО ЗАПУСКА PIPELINE!
MAX_LENGTH = 200
vocab = ...  # загрузи словарь как раньше

# --- Пример: как теперь выглядит say_message ---
def say_message(msg):
    # больше не вызываем vocalise() напрямую!
    # вместо этого — кладём в очередь
    text_queue.put(msg)

# --- Главный асинхронный агент ---
async def run_agent_aysnc():
    print("📢 Приветствую...")
    say_message(WELCOME_MESSAGE)  # ← отправляем первое сообщение в очередь

    print("🚀 Запускаю конвейер...")
    pipeline_task = asyncio.create_task(start_pipeline())

    print("⏳ Жду, пока приветствие проиграется...")
    await audio_buffer.join()  # ← ТУТ ДОЛЖНО БЫТЬ ПАУЗА!
    #await asyncio.sleep(1.0)
    print("✅ Приветствие проиграно! Теперь можно слушать.")

    try:
        while True:
            print("\n--- Ожидаю речи... ---")
            inp = await listen_and_recognize_phrase(timeout=15.0)

            if not inp.strip():
                print("🔇 Пустой ввод, пропускаем...")
                continue

            print(f"📩 Отправляем запрос: {inp}")
            await asyncio.to_thread(send_chat_request_queued, inp, True, text_queue)

            await audio_buffer.join()  # ← ТУТ ДОЛЖНО БЫТЬ ПАУЗА!
            #await asyncio.sleep(1.0)

    except KeyboardInterrupt:
        print("\n🛑 Остановка агента...")
        stop_pipeline()
        await pipeline_task
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        stop_pipeline()
        await pipeline_task

# --- Запуск ---
if __name__ == "__main__":
    try:
        asyncio.run(run_agent_aysnc())
    except KeyboardInterrupt:
        print("\n👋 Программа остановлена пользователем.")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")