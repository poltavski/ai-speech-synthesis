import json
import torch
import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from utils import apply_tts, init_config, init_models, init_jit_model

# from uuid import uuid4
from datetime import datetime

import numpy as np
import soundfile as sf


app = FastAPI()
device = torch.device("cuda")
models = init_models(device=device)
vocab, rate, output_dir = init_config()


@app.get("/")
def index():
    """Index route."""
    return "Hello from text to speech server."


@app.post("/speech")
def speech(text: str = Body("", embed=True), voice: str = "Dina"):
    """Speech route."""
    if not text:
        text = "Макс+им Валент+инович! М+ы озв+учиваем в+ашу р+ечь. А собр+али н+ас прил+ежные студ+енты, Полт+авский и Мезг+а"
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
