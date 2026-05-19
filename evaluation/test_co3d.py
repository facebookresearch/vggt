import os
import torch
import numpy as np
import gzip
import json
import random
import logging
import warnings
from vggt.models.vggt import VGGT
from vggt.models.vggt_small import VGGT as VGGTsmall
from vggt.utils.rotation import mat_to_quat
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3
import argparse
# python test_co3d.py --model_path ../pretrained_models/model_tracker_fixed_e20.pt --co3d_dir /mimer/NOBACKUP/groups/3d-dl/co3dv2 co3d_anno_dir ../annotations/co3d_v2_annotations
# python test_co3d.py --model_path ../training/logs/dinov3_exp004/ckpts/checkpoint.pt --co3d_dir /mimer/NOBACKUP/groups/3d-dl/co3dv2 co3d_anno_dir ../annotations/co3d_v2_annotations --encoder dinov3


# Suppress DINO v2 logs
logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="dinov2")

# Set computation precision
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False


def convert_pt3d_RT_to_opencv(Rot, Trans):
    """
    Convert Point3D extrinsic matrices to OpenCV convention.

    Args:
        Rot: 3D rotation matrix in Point3D format
        Trans: 3D translation vector in Point3D format

    Returns:
        extri_opencv: 3x4 extrinsic matrix in OpenCV format
    """
    rot_pt3d = np.array(Rot)
    trans_pt3d = np.array(Trans)

    trans_pt3d[:2] *= -1
    rot_pt3d[:, :2] *= -1
    rot_pt3d = rot_pt3d.transpose(1, 0)
    extri_opencv = np.hstack((rot_pt3d, trans_pt3d[:, None]))
    return extri_opencv


def build_pair_index(N, B=1):
    """
    Build indices for all possible pairs of frames.

    Args:
        N: Number of frames
        B: Batch size

    Returns:
        i1, i2: Indices for all possible pairs
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using NumPy.

    Args:
        r_error: numpy array representing R error values (Degree)
        t_error: numpy array representing T error values (Degree)
        max_threshold: Maximum threshold value for binning the histogram

    Returns:
        AUC value and the normalized histogram
    """
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.
    This function assumes the input poses are world-to-camera (w2c) transformations.

    Args:
        pred_se3: Predicted SE(3) transformations (w2c), shape (N, 4, 4)
        gt_se3: Ground truth SE(3) transformations (w2c), shape (N, 4, 4)
        num_frames: Number of frames (N)

    Returns:
        Rotation and translation angle errors in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    relative_pose_gt = gt_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(gt_se3[pair_idx_i2])
    )
    relative_pose_pred = pred_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(pred_se3[pair_idx_i2])
    )

    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg


def setup_args():
    """Set up command-line arguments for the CO3D evaluation script."""
    parser = argparse.ArgumentParser(description='Test VGGT on CO3D dataset')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (only test on specific category)')
    parser.add_argument('--fast_eval', action='store_true', default=False, help='Only evaluate 10 sequences per category')
    
    parser.add_argument('--big_model', action='store_true', default=False, help='If to load the original VGGT')
    parser.add_argument('--encoder', type=str, default="dinov3", help='Encoder to use in VGGTsmall')
    
    parser.add_argument('--min_num_images', type=int, default=50, help='Minimum number of images for a sequence')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to use for testing')
    parser.add_argument('--co3d_dir', type=str, required=True, help='Path to CO3D dataset')
    parser.add_argument('--co3d_anno_dir', type=str, required=True, help='Path to CO3D annotations')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the VGGT model checkpoint')
    return parser.parse_args()


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
    else:
        model = VGGT()
    print(f"USING {model_path}")
    model.load_state_dict(torch.load(model_path)['model'], strict=True)
    model.eval()
    model = model.to(device)
    return model



def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def process_sequence(model, seq_name, seq_data, category, co3d_dir, min_num_images, num_frames, device, dtype):
    """
    Process a single sequence and compute pose errors.

    Args:
        model: VGGT model
        seq_name: Sequence name
        seq_data: Sequence data
        category: Category name
        co3d_dir: CO3D dataset directory
        min_num_images: Minimum number of images required
        num_frames: Number of frames to sample
        device: Device to run on
        dtype: Data type for model inference

    Returns:
        rError: Rotation errors
        tError: Translation errors
    """
    if len(seq_data) < min_num_images:
        return None, None

    metadata = []
    for data in seq_data:
        # Make sure translations are not ridiculous
        if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
            return None, None
        extri_opencv = convert_pt3d_RT_to_opencv(data["R"], data["T"])
        metadata.append({
            "filepath": data["filepath"],
            "extri": extri_opencv,
        })

    # Random sample num_frames images
    ids = np.random.choice(len(metadata), num_frames, replace=False)
    print("Image ids", ids)

    image_names = [os.path.join(co3d_dir, metadata[i]["filepath"]) for i in ids]
    gt_extri = [np.array(metadata[i]["extri"]) for i in ids]
    gt_extri = np.stack(gt_extri, axis=0)

    images = load_and_preprocess_images(image_names).to(device)


    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    with torch.cuda.amp.autocast(dtype=torch.float64):
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        pred_extrinsic = extrinsic[0]

    with torch.cuda.amp.autocast(dtype=torch.float64):
        gt_extrinsic = torch.from_numpy(gt_extri).to(device)
        add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4)

        pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
        gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)


        Racc_5 = (rel_rangle_deg < 5).float().mean().item()
        Tacc_5 = (rel_tangle_deg < 5).float().mean().item()

        print(f"{category} sequence {seq_name} R_ACC@5: {Racc_5:.4f}")
        print(f"{category} sequence {seq_name} T_ACC@5: {Tacc_5:.4f}")

        return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy()


def main():
    """Main function to evaluate VGGT on CO3D dataset."""
    # Parse command-line arguments
    args = setup_args()

    # Setup device and data type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Load model
    model = load_model(device, model_path=args.model_path, big_model=args.big_model, encoder=args.encoder)

    # Set random seeds
    set_random_seeds(args.seed)

    # Categories to evaluate
    SEEN_CATEGORIES = [
        "apple",
        "bench",
        "bowl", "cellphone", "frisbee", "hotdog",
        # "keyboard",
        "parkingmeter",
        "teddybear",
        "toybus",
        "backpack",
        "book",
        "car",
        "donut",
        "handbag",
        "hydrant",
        "motorcycle",
        "pizza",
        "stopsign",
        "toaster",
        "tv"
    ]

    if args.debug:
        SEEN_CATEGORIES = ["parkingmeter"]

    per_category_results = {}

    for category in SEEN_CATEGORIES:
        print(f"Loading annotation for {category} test set")
        annotation_file = os.path.join(args.co3d_anno_dir, f"{category}_test.jgz")

        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            print(f"Annotation file not found for {category}, skipping")
            continue

        rError = []
        tError = []

        seq_names = sorted(list(annotation.keys()))
        if args.fast_eval and len(seq_names)>=10:
            seq_names = random.sample(seq_names, 10)
        seq_names = sorted(seq_names)


        print("Testing Sequences: ")
        print(seq_names)

        for seq_name in seq_names:
            seq_data = annotation[seq_name]
            print("-" * 50)
            print(f"Processing {seq_name} for {category} test set")
            if args.debug and not os.path.exists(os.path.join(args.co3d_dir, category, seq_name)):
                print(f"Skipping {seq_name} (not found)")
                continue

            try: 
                seq_rError, seq_tError = process_sequence(
                    model, seq_name, seq_data, category, args.co3d_dir,
                    args.min_num_images, args.num_frames, device, dtype,
                )
            except Exception as e:
                print(f"Error processing {seq_name}: {e}")
                continue

            print("-" * 50)

            if seq_rError is not None and seq_tError is not None:
                rError.extend(seq_rError)
                tError.extend(seq_tError)

        if not rError:
            print(f"No valid sequences found for {category}, skipping")
            continue

        rError = np.array(rError)
        tError = np.array(tError)

        Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
        Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
        Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
        Auc_3, _ = calculate_auc_np(rError, tError, max_threshold=3)

        per_category_results[category] = {
            "rError": rError,
            "tError": tError,
            "Auc_30": Auc_30,
            "Auc_15": Auc_15,
            "Auc_5": Auc_5,
            "Auc_3": Auc_3
        }

        print("="*80)
        # Print results with colors
        GREEN = "\033[92m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        print(f"{BOLD}{BLUE}AUC of {category} test set:{RESET} {GREEN}{Auc_30:.4f} (AUC@30), {Auc_15:.4f} (AUC@15), {Auc_5:.4f} (AUC@5), {Auc_3:.4f} (AUC@3){RESET}")
        mean_AUC_30_by_now = np.mean([per_category_results[category]["Auc_30"] for category in per_category_results])
        mean_AUC_15_by_now = np.mean([per_category_results[category]["Auc_15"] for category in per_category_results])
        mean_AUC_5_by_now = np.mean([per_category_results[category]["Auc_5"] for category in per_category_results])
        mean_AUC_3_by_now = np.mean([per_category_results[category]["Auc_3"] for category in per_category_results])
        print(f"{BOLD}{BLUE}Mean AUC of categories by now:{RESET} {RED}{mean_AUC_30_by_now:.4f} (AUC@30), {mean_AUC_15_by_now:.4f} (AUC@15), {mean_AUC_5_by_now:.4f} (AUC@5), {mean_AUC_3_by_now:.4f} (AUC@3){RESET}")
        print("="*80)

    # Print summary results
    print("\nSummary of AUC results:")
    print("-"*50)
    for category in sorted(per_category_results.keys()):
        print(f"{category:<15}: {per_category_results[category]['Auc_30']:.4f} (AUC@30), {per_category_results[category]['Auc_15']:.4f} (AUC@15), {per_category_results[category]['Auc_5']:.4f} (AUC@5), {per_category_results[category]['Auc_3']:.4f} (AUC@3)")

    if per_category_results:
        mean_AUC_30 = np.mean([per_category_results[category]["Auc_30"] for category in per_category_results])
        mean_AUC_15 = np.mean([per_category_results[category]["Auc_15"] for category in per_category_results])
        mean_AUC_5 = np.mean([per_category_results[category]["Auc_5"] for category in per_category_results])
        mean_AUC_3 = np.mean([per_category_results[category]["Auc_3"] for category in per_category_results])
        print("-"*50)
        print(f"Mean AUC: {mean_AUC_30:.4f} (AUC@30), {mean_AUC_15:.4f} (AUC@15), {mean_AUC_5:.4f} (AUC@5), {mean_AUC_3:.4f} (AUC@3)")
    print(args.model_path)

if __name__ == "__main__":
    main()