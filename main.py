import json
import torch
import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from utils import apply_tts, init_config, init_models, init_jit_model
from stressrnn import StressRNN

# from uuid import uuid4
from datetime import datetime

import numpy as np
import soundfile as sf
import requests

app = FastAPI()
device = torch.device("cuda")
models = init_models(device=device)
vocab, rate, output_dir = init_config()
stress_rnn = StressRNN()


@app.get("/")
def index():
    """Index route."""
    return "Hello from text to speech server."


@app.post("/speech")
def speech(text: str = Body("", embed=True), voice: str = "Dina", stress: bool = True, chatbot: bool = False):
    """Speech route."""
    try:
        if not text:
            text = "Максим Валентинович! Мы озвучиваем вашу речь. А собрали нас прилежные студенты, Полтавский и Мезга"
        if chatbot:
            data = {
                "uid":"cc522292-1098-4e2a-89f7-68b49f7f35b6",
                "bot":"main",
                "text": text,
            }
            headers = {'cookie': '_ym_uid=1621870280388693023; _ym_d=1621870280; _ga=GA1.2.344477183.1621870280; _gid=GA1.2.1036211673.1621870280; _ym_isad=1; SL_GWPT_Show_Hide_tmp=1; SL_wptGlobTipTmp=1; _xbs_pp=1621870292671'}
            r = requests.post('https://xu.su/api/send', data=data, headers=headers)
            answer = json.loads(r.text)
            text = answer["text"] if answer.get("text") else text
        if stress:
            text = stress_rnn.put_stress(
                text,
                stress_symbol="+",
                accuracy_threshold=0.75,
                replace_similar_symbols=True,
            )
        audio = apply_tts(
            texts=[text],
            model=models[voice.lower()],
            sample_rate=rate,
            symbols=vocab,
            device=device,
        )
        # filename = f"{voice}_{uuid4()}.wav"
        date = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
        filename = f"{output_dir}/{voice}_{date}.wav"
        sf.write(filename, audio[0].data.cpu().numpy().astype(np.float32), rate)
        return FileResponse(filename)
    except:
        return FileResponse(f"{output_dir}/default.wav")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
