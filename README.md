# orin
Local voice assistant on rockchip platform

asr and tts models - included.
For llm - start flask server like `python3 flask_server.py --target_platform rk3588 --rkllm_model_path ~/Repo/qwen/Qwen3-1.7B-rk3588-w8a8.rkllm`
The model can be downloaded from rknn-llm package
then just `python3 orin_qwen.py` 
Audio devices hardcoded as mic from 3.5 jack and usb speaker.
