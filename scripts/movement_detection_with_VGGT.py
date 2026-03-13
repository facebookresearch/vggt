# For mps on Mac
# import os
# # enable CPU fallback for unsupported MPS ops; must be set before importing torch
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# import sys
# print(sys.executable)
# import torch
# print(torch.__version__)


import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


device = "cuda" if torch.cuda.is_available() else \
         "mps"  if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"Device: {device}, dtype: {dtype}")


# model download and loading
PYTORCH_ENABLE_MPS_FALLBACK=1 # fallback to cpu

model = VGGT.from_pretrained("facebook/vggt-1b")
model = model.to(device=device, dtype=dtype)
model.eval()

# load and preprocess images
# do inference of your images at once if you want to compare the poses in the same coordinate system
images = load_and_preprocess_images(["539_latest.jpg", "539_reference.jpg"]).to(device)

with torch.no_grad():
    with torch.autocast(device_type=device if device != "mps" else "cpu", dtype=dtype):
        predictions = model(images)


# Image dimesions processed by VGGT. Got them from: for key, val in predictions.items():
image_size_hw = (294, 518)

# Decode pose_enc
extrinsics, intrinsics = pose_encoding_to_extri_intri(
    predictions["pose_enc"], 
    image_size_hw=image_size_hw
)
print("Extrinsics shape:", extrinsics.shape)

# 2 cameras extractions — shape [3, 4] each
cam1 = extrinsics[0, 0].cpu().float().numpy()  # [batch=0, img=0]
cam2 = extrinsics[0, 1].cpu().float().numpy()  # [batch=0, img=1]

# To create rotation and translation from matrix:
#         columns -> 0  1  2  3
#         row 0    [ r  r  r  tx ]
#         row 1    [ r  r  r  ty ]
#         row 2    [ r  r  r  tz ]
R1, t1 = cam1[:3, :3], cam1[:3, 3]
R2, t2 = cam2[:3, :3], cam2[:3, 3]

# Relative transformation between camera 1 and camera 2
R_rel = R2 @ R1.T # I transpose R1 because the matrix is orthogonal, so R^-1 = R^T
t_rel = t2 - R_rel @ t1


angle_deg = np.linalg.norm(R.from_matrix(R_rel).as_rotvec()) * 180 / np.pi # angle in degrees from rotation vector

trans_norm = np.linalg.norm(t_rel) # euclidean distance of the translation vector

print(f"\nTranslation: {trans_norm:.4f}")
print(f"Rotation:   {angle_deg:.2f}°")

# Adjust parameters
TRANS_THR = 0.05
ROT_THR   = 2.0

if trans_norm > TRANS_THR or angle_deg > ROT_THR:
    print("Camera moved!")
else:
    print("Camera OK")