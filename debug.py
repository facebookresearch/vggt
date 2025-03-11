import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import os
import socket
import webbrowser
sys.path.append('vggt/')
import shutil
from datetime import datetime
from demo_hf import demo_fn #, initialize_model
from omegaconf import DictConfig, OmegaConf
import glob
import gc
import time
from viser_fn import viser_wrapper
from gradio_util import demo_predictions_to_glb
from hydra.utils import instantiate
# import spaces



print("Loading model")

cfg_file = "config/base.yaml"
cfg = OmegaConf.load(cfg_file)
vggt_model = instantiate(cfg, _recursive_=False)


