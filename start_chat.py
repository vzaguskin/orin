from mms_tts import TTSVocaliser
from vosk_recognizer_async import create_recognizer, listen_and_recognize_phrase

import asyncio
from queue import Queue as SyncQueue  # Для передачи данных в синхронный код (если нужно)
import sys
import queue

import threading

# Глобальный флаг для остановки
is_running = True

text_queue = queue.Queue()

WELCOME_MESSAGE = "Привет, я Орин"
voc = TTSVocaliser()
rec = create_recognizer()

async def audio_player_task():
    """Асинхронный таск, который читает текст из синхронной очереди и проигрывает его"""
    global is_running

    print("🔊 Запущен таск проигрывания аудио...")

    while is_running:
        try:
            # Асинхронно ждём текст из синхронной очереди
            text = await asyncio.to_thread(text_queue.get, timeout=0.5)  # timeout — чтобы не висеть навсегда

            # Проверяем сигнал остановки
            if text is None:
                print("🛑 Получен сигнал остановки audio_player_task")
                break

            print(f"🎧 Проигрывание: '{text}'")
            # Блокирующий вызов — но он запущен в to_thread? Нет! Он будет выполнен в основном потоке.
            # Но vocalise — блокирующий, и мы его вызываем здесь — это нормально!
            # Почему? Потому что мы НЕ хотим параллельно синтезировать много аудио — мы хотим **последовательное** проигрывание.
            # Т.е. пока играет одно — следующее ждёт в очереди.
            voc.vocalise(text)  # ← Это синхронный, блокирующий вызов — но это ОК!

        except queue.Empty:
            # Таймаут — ничего не пришло, продолжаем ждать
            continue
        except Exception as e:
            print(f"❌ Ошибка в audio_player_task: {e}")
            # Не ломаем цикл — просто пропускаем битый фрагмент
            continue

    print("✅ audio_player_task завершён.")


from llm_chat import send_chat_request_queued
def say_message(msg):
    voc.vocalise(msg)

def get_chat_reply(inp):
    return inp

def get_user_input():
    return listen_and_recognize_phrase()

def run_agent():
    say_message(WELCOME_MESSAGE)
    while True:
        inp = get_user_input()
        #rep = get_chat_reply(inp)
        #say_message(rep)
        send_chat_request(inp, voice_callback=say_message)
        #send_chat_request(inp, voice_callback=None)

async def run_agent_aysnc():
    say_message(WELCOME_MESSAGE)

    # Запускаем таск проигрывания аудио
    player_task = asyncio.create_task(audio_player_task())

    try:
        while True:
            print("\n--- Ожидаю речи... ---")
            inp = await listen_and_recognize_phrase(timeout=15.0)

            if not inp.strip():
                print("🔇 Пустой ввод, пропускаем...")
                continue

            print(f"📩 Отправляем запрос: {inp}")
            # send_chat_request — пока синхронная, но мы её тоже запускаем в потоке
            await asyncio.to_thread(send_chat_request_queued, inp, True, text_queue)

    except KeyboardInterrupt:
        print("\n🛑 Остановка агента...")
        await text_queue.put(None)
        await player_task
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        await text_queue.put(None)
        await player_task

def main():
    run_agent()

if __name__ == "__main__":
    #main()
    try:
        asyncio.run(run_agent_aysnc())
    except KeyboardInterrupt:
        print("\n👋 Программа остановлена пользователем.")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")