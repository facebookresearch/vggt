import os
import shutil
import pickle
from threading import Thread
import gc

from fastapi import FastAPI, Request, Response
import numpy as np
from pyngrok import ngrok
import uvicorn
from PIL import Image

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import preprocess_images
from sim.pointclouds.inference import CudaInference

if not torch.cuda.is_available():
    raise ValueError("CUDA is not available. Check your environment.")
device = "cuda"

print("Loading model")
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model.eval()
model.to(device)
print("Finished loading model")

app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    body = await request.body()
    imgs = pickle.loads(body)
    predictions = run_model(imgs)
    response = Response(
        content = pickle.dumps(predictions),
        media_type = "application/octet-stream"
    )
    return response


def run_model(images: list) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    images = preprocess_images(images).to(device)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    
    torch.cuda.empty_cache()
    return predictions


def start_api():
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == '__main__':
    Thread(target=start_api, daemon=True).start()

    public_url = ngrok.connect(8000)
    print(f"Public API endpoint: {public_url}/predict")

    input("Press Enter to exit...\n")
