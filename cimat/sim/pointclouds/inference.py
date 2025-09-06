import os
import glob
import time
import gc
import pickle

import torch
import numpy as np
import requests

from vggt.models.vggt import VGGT
from visual_util import predictions_to_glb
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

#Fix this messs. This code is the ugliest you have developed in months

device = "cuda"

class CudaInference:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Check your environment.")
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        self.model.eval()
        self.model.to(device)

    def clear_model(self) -> None:
        del self.model
        self.model = None

    def run(self, prediction_data: object) -> object:
        if self.model is None:
            raise ValueError("Model must be loaded first.")
        target_dir = prediction_data.target_dir
        gc.collect()
        torch.cuda.empty_cache()

        image_names = [f"{target_dir}/images/{name}" for name in prediction_data.image_names]

        images = load_and_preprocess_images(image_names).to(device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = self.model(images)
        
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Clean up
        torch.cuda.empty_cache()

        # Convert tensors to numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

        predictions["image_names"] = prediction_data.image_names
        depth_map = predictions["depth"]  # (S, H, W, 1)
        world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        predictions["world_points_from_depth"] = world_points

        prediction_save_path = os.path.join(target_dir, "predictions.npz")
        np.savez(prediction_save_path, **predictions)
        prediction_data._predictions = predictions

        return prediction_data


class ApiInference:
    def __init__(self, url: str) -> None:
        self.url = url

    def run(self, prediction_data: object) -> object:
        target_dir = prediction_data.target_dir
        image_names = [
            f"{target_dir}/images/{name}"
            for name in prediction_data.image_names
        ]
