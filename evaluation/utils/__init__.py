from vggt.models.vggt_small import VGGT as VGGTsmall
from vggt.models.vggt import VGGT
import torch

def load_model(device, model_path, big_model=False, encoder="dinov3"):
    """
    Load the VGGT model.

    Args:
        device: Device to load the model on
        model_path: Path to the model checkpoint

    Returns:
        Loaded VGGT model
    """
    print("Initializing and loading VGGT model...")
    if not big_model:
        model = VGGTsmall(
            img_size=336,
            embed_dim=768,
            depth=6,
            num_heads=12,
            patch_size=16,
            patch_embed=encoder,
            enable_camera=True,
            enable_depth=True,
            enable_point=True,
            enable_track=False,
        )
        state_dict = torch.load(model_path)['model']
    else:
        model = VGGT()
        state_dict = torch.load(model_path)
    print(f"USING {model_path}")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)
    return model