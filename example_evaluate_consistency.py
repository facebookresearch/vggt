"""
Example script demonstrating how to use the geometry consistency evaluation.

This script shows various ways to use the evaluation functions programmatically.
"""

import torch
from pathlib import Path
from evaluate_geometry_consistency_fast import evaluate_geometry_consistency_fast

def example_basic_usage():
    """
    Basic usage example: Evaluate a single folder of images.
    """
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60)

    # Path to your folder containing object segment images
    input_folder = "path/to/your/object/segments"

    # Run evaluation with default parameters (32 frames, max_size=518)
    results = evaluate_geometry_consistency_fast(
        input_folder=input_folder,
        num_frames=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )

    print(f"\nResults:")
    print(f"  Average PSNR: {results['avg_psnr']:.2f} dB")
    print(f"  Average SSIM: {results['avg_ssim']:.4f}")


def example_custom_parameters():
    """
    Example with custom parameters.
    """
    print("\n" + "="*60)
    print("Example 2: Custom Parameters")
    print("="*60)

    input_folder = "path/to/your/object/segments"

    # Run evaluation with custom parameters
    results = evaluate_geometry_consistency_fast(
        input_folder=input_folder,
        num_frames=64,  # Sample more frames
        max_size=448,   # Use smaller image size for faster processing
        device='cuda',
        use_pytorch3d=True,  # Use PyTorch3D for fast rendering
        verbose=True
    )

    # Access per-frame metrics
    print("\nPer-frame metrics:")
    for i, (psnr, ssim) in enumerate(zip(results['psnr_per_frame'], results['ssim_per_frame'])):
        print(f"  Frame {i}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")


def example_batch_evaluation():
    """
    Example: Evaluate multiple folders in batch.
    """
    print("\n" + "="*60)
    print("Example 3: Batch Evaluation")
    print("="*60)

    # List of folders to evaluate
    folders = [
        "path/to/object1",
        "path/to/object2",
        "path/to/object3",
    ]

    all_results = {}

    for folder in folders:
        print(f"\nEvaluating: {folder}")
        try:
            results = evaluate_geometry_consistency_fast(
                input_folder=folder,
                num_frames=32,
                device='cuda',
                verbose=False  # Suppress per-folder output
            )
            all_results[folder] = results
            print(f"  PSNR: {results['avg_psnr']:.2f} dB, SSIM: {results['avg_ssim']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    avg_psnr = sum(r['avg_psnr'] for r in all_results.values()) / len(all_results)
    avg_ssim = sum(r['avg_ssim'] for r in all_results.values()) / len(all_results)
    print(f"Overall Average PSNR: {avg_psnr:.2f} dB")
    print(f"Overall Average SSIM: {avg_ssim:.4f}")


def example_using_base_version():
    """
    Example: Using the base version without PyTorch3D.
    """
    print("\n" + "="*60)
    print("Example 4: Using Base Version (No PyTorch3D)")
    print("="*60)

    from evaluate_geometry_consistency import evaluate_geometry_consistency

    input_folder = "path/to/your/object/segments"

    # This version doesn't require PyTorch3D but is slower
    results = evaluate_geometry_consistency(
        input_folder=input_folder,
        num_frames=32,
        device='cuda',
        verbose=True
    )

    print(f"\nResults:")
    print(f"  Average PSNR: {results['avg_psnr']:.2f} dB")
    print(f"  Average SSIM: {results['avg_ssim']:.4f}")


def example_save_and_load_results():
    """
    Example: Save and load results from JSON.
    """
    print("\n" + "="*60)
    print("Example 5: Save and Load Results")
    print("="*60)

    import json

    input_folder = "path/to/your/object/segments"

    # Run evaluation
    results = evaluate_geometry_consistency_fast(
        input_folder=input_folder,
        num_frames=32,
        device='cuda',
        verbose=False
    )

    # Save results
    output_file = Path(input_folder) / 'geometry_consistency_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")

    # Load results
    with open(output_file, 'r') as f:
        loaded_results = json.load(f)

    print(f"\nLoaded results:")
    print(f"  Average PSNR: {loaded_results['avg_psnr']:.2f} dB")
    print(f"  Average SSIM: {loaded_results['avg_ssim']:.4f}")
    print(f"  Number of frames: {loaded_results['num_frames']}")


def example_custom_workflow():
    """
    Example: Custom workflow with manual control over each step.
    """
    print("\n" + "="*60)
    print("Example 6: Custom Workflow")
    print("="*60)

    import glob
    from evaluate_geometry_consistency import (
        uniform_sample_frames,
        load_and_preprocess_segments,
        save_preprocessed_images,
        run_vggt_inference,
        reconstruct_point_cloud,
        reproject_point_cloud,
        save_rendered_images,
        compute_psnr,
        compute_ssim,
    )
    from vggt.models.vggt import VGGT

    input_folder = Path("path/to/your/object/segments")
    device = 'cuda'

    # Step 1: Sample images
    image_paths = sorted(glob.glob(str(input_folder / '*.jpg')))
    sampled_paths = uniform_sample_frames(image_paths, num_frames=32)
    print(f"Sampled {len(sampled_paths)} frames")

    # Step 2: Preprocess
    images, metadata_list = load_and_preprocess_segments(sampled_paths, max_size=518)
    print(f"Preprocessed images: {images.shape}")

    # Step 3: Save cached images
    cache_folder = input_folder / 'vggt_cache'
    save_preprocessed_images(images, sampled_paths, cache_folder)
    print(f"Saved to: {cache_folder}")

    # Step 4: Load VGGT model
    model = VGGT.from_pretrained("facebook/vggt")
    model = model.to(device).eval()
    print("Model loaded")

    # Step 5: Run inference
    predictions = run_vggt_inference(model, images, device=device)
    print(f"VGGT inference complete")

    # Step 6: Reconstruct point cloud
    points, colors = reconstruct_point_cloud(
        predictions['depth'],
        predictions['extrinsic'],
        predictions['intrinsic'],
        images,
        metadata_list
    )
    print(f"Point cloud: {len(points)} points")

    # Step 7: Reproject
    H, W = images.shape[-2:]
    rendered_images, rendered_depths = reproject_point_cloud(
        points, colors,
        predictions['extrinsic'],
        predictions['intrinsic'],
        (H, W),
        device=device
    )
    print(f"Reprojection complete")

    # Step 8: Save rendered images
    rerender_folder = input_folder / 'vggt_rerender'
    save_rendered_images(rendered_images, sampled_paths, rerender_folder)
    print(f"Saved to: {rerender_folder}")

    # Step 9: Compute metrics
    psnr_values = []
    ssim_values = []
    for i in range(len(rendered_images)):
        cached_img = images[i].permute(1, 2, 0).to(device)
        rendered_img = rendered_images[i]
        psnr = compute_psnr(cached_img, rendered_img)
        ssim = compute_ssim(cached_img, rendered_img)
        psnr_values.append(psnr)
        ssim_values.append(ssim)

    import numpy as np
    avg_psnr = np.mean([p for p in psnr_values if p != float('inf')])
    avg_ssim = np.mean(ssim_values)

    print(f"\nResults:")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average SSIM: {avg_ssim:.4f}")


if __name__ == '__main__':
    print("Geometry Consistency Evaluation - Example Usage")
    print("="*60)
    print("\nNote: Replace 'path/to/your/object/segments' with actual paths")
    print("      before running these examples.\n")

    # Uncomment the example you want to run:

    # example_basic_usage()
    # example_custom_parameters()
    # example_batch_evaluation()
    # example_using_base_version()
    # example_save_and_load_results()
    # example_custom_workflow()

    print("\nTo run an example, uncomment the corresponding function call in main.")
