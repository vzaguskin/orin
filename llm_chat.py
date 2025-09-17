import sys
import requests
import json
import re

# Set the address of the Server.
server_url = 'http://192.168.1.17:8080/rkllm_chat'
MODEL = "Qwen3-0.6B-rk3588-w8a8.rkllm"
MODEL_DIR = "/home/pi/Repo/qwen"

# Create a session object.
session = requests.Session()
session.keep_alive = False  # Close the connection pool to maintain a long connection.
adapter = requests.adapters.HTTPAdapter(max_retries=5)
session.mount('https://', adapter)
session.mount('http://', adapter)

MESSAGES = [{"role": "system", "content": "Ты дружелюбный ассистент по имени Орин. Разговаривай по русски. Пиши все латинские слова и цифры русскими буквами, например вместо Samsung пиши Самсунг, вместо 125 - сто двадцать пять."}]
PROMPT = "Ты дружелюбный ассистент. Пиши ответ русскими буквами без цифр и эмоджи. Вопрос: "

PROMPT = "Ты дружелюбный ассистент по имени Орин. Разговаривай по русски. Пиши все латинские слова и цифры русскими буквами, например вместо Samsung пиши Самсунг, вместо 125 - сто двадцать пять. Вопрос: "
def send_chat_request(user_message, is_streaming=True, voice_callback = None):
    print("Поступил запрос:", user_message)
    headers = {
                    'Content-Type': 'application/json',
                    'Authorization': 'not_required'
                }
    data = {
                    "model": MODEL,
                    "messages": [{"role": "user", "content":  user_message}],
                    "stream": is_streaming,
                    "enable_thinking": False,
                    "tools": None
                }
    print("Data:", data)
    # Send a POST request
    responses = session.post(server_url, json=data, headers=headers, stream=is_streaming, verify=False)

    if not is_streaming:
        # Parse the response
        if responses.status_code == 200:
            print("Q:", data["messages"][-1]["content"])
            reply = json.loads(responses.text)["choices"][-1]["message"]["content"]
            print("A:", reply)
            if voice_callback:
                voice_callback(reply)
        else:
            print("Error:", responses.text)
    else:
        if responses.status_code == 200:
            print("Q:", data["messages"][-1]["content"])
            print("A:", end="")
            buff = ""
            for line in responses.iter_lines():
                if line:
                    line = json.loads(line.decode('utf-8'))
                    if line["choices"][-1]["finish_reason"] != "stop":
                        reply = line["choices"][-1]["delta"]["content"]
                        #print("reply:", reply, end="")
                        buff += reply
                        sys.stdout.flush()
                        if len(buff) > 200 or buff[-1] in ['.', '?', '!']:
                            print("озвучиваю:", buff)
                            if voice_callback:
                                voice_callback(buff)
                            
                            buff = ""
            if buff and voice_callback:
                voice_callback(buff)
                print("озвучиваю:", buff)
                        
                            
        else:
            print('Error:', responses.text)


def send_chat_request_queued(user_message, is_streaming=True, voice_queue = None):
    print("Поступил запрос:", user_message)
    headers = {
                    'Content-Type': 'application/json',
                    'Authorization': 'not_required'
                }
    data = {
                    "model": MODEL,
                    "messages": [{"role": "user", "content":  user_message}],
                    "stream": is_streaming,
                    "enable_thinking": False,
                    "tools": None
                }
    print("Data:", data)
    # Send a POST request
    responses = session.post(server_url, json=data, headers=headers, stream=is_streaming, verify=False)

    if not is_streaming:
        # Parse the response
        if responses.status_code == 200:
            print("Q:", data["messages"][-1]["content"])
            reply = json.loads(responses.text)["choices"][-1]["message"]["content"]
            print("A:", reply)
            if voice_callback:
                voice_callback(reply)
        else:
            print("Error:", responses.text)
    else:
        if responses.status_code == 200:
            print("Q:", data["messages"][-1]["content"])
            print("A:", end="")
            buff = ""
            for line in responses.iter_lines():
                if line:
                    line = json.loads(line.decode('utf-8'))
                    if line["choices"][-1]["finish_reason"] != "stop":
                        reply = line["choices"][-1]["delta"]["content"]
                        #print("reply:", reply, end="")
                        buff += reply
                        sys.stdout.flush()
                        if len(buff) > 200 or buff[-1] in ['.', '?', '!']:
                            print("озвучиваю:", buff)
                            if voice_queue:
                                voice_queue.put(buff)
                            
                            buff = ""
            if buff and voice_queue:
                voice_queue.put(buff)
                print("озвучиваю:", buff)
                        
                            
        else:
            print('Error:', responses.text)
