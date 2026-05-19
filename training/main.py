import torch
from hydra import initialize, compose
from hydra.utils import instantiate
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np

def save_ply(points, colors, filename):
    import open3d as o3d                
    if torch.is_tensor(points):
        points_visual = points.reshape(-1, 3).cpu().numpy()
    else:
        points_visual = points.reshape(-1, 3)
    if torch.is_tensor(colors):
        points_visual_rgb = colors.reshape(-1, 3).cpu().numpy()
    else:
        points_visual_rgb = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_visual.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(points_visual_rgb.astype(np.float64))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name="default")

dist.init_process_group(
    backend="nccl",
)

train_dataset = instantiate(cfg.data.train, _recursive_=False)
train_dataset.seed = 1337

dataloader = train_dataset.get_loader(epoch=0)

create_ply = True

for i, batch in enumerate(dataloader):
    if create_ply: 
        save_ply(
            batch["world_points"][0].reshape(-1, 3), 
            batch["images"][0].permute(0, 2, 3, 1).reshape(-1, 3), 
            f"debug_{i:04d}.ply"
        )
        print(f"Saved debug_{i:04d}.ply")
    else:
        images = batch['images']  # [B, T, C, H, W]
        depths = batch['depths']  # [B, T, H, W]

        # pick first sample and first frame
        img = images[0, 0]        # [C, H, W]
        depth = depths[0, 0]      # [H, W]

        # move channels last for matplotlib
        img_np = img.permute(1, 2, 0).cpu().numpy()
        depth_np = depth.cpu().numpy()

        # normalize depth for visualization
        depth_vis = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)

        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.imshow(img_np)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(depth_vis, cmap="plasma")  # "viridis", "magma" also nice
        plt.title("Depth")
        plt.axis("off")

        plt.savefig(f"sample_{i:04d}.png")
        plt.close()

        print(f"Saved sample_{i:04d}.png")