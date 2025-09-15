from mms_tts import TTSVocaliser
from vosk_recogniser import create_recognizer, listen_and_recognize_phrase
WELCOME_MESSAGE = "Привет, я Орин"
voc = TTSVocaliser()
rec = create_recognizer()
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
        rep = get_chat_reply(inp)
        say_message(rep)



def main():
    run_agent()

if __name__ == "__main__":
    main()