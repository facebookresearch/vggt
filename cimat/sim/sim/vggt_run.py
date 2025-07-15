import os
import glob
import time
import gc

import torch
import numpy as np

from vggt.models.vggt import VGGT
from visual_util import predictions_to_glb
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


class VggtInference:
    def __init__(self, cpu: bool = False) -> None:
        model = 