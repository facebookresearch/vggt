import hydra
import torch
import os
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms as TF
import glob
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from viser_fn import viser_wrapper
from vggt.utils.geometry import depth_to_world_coords_points
from omegaconf import DictConfig, OmegaConf

# @hydra.main(config_path="config", config_name="base")
def demo_fn(cfg: DictConfig, model) -> None:
    print(cfg.SCENE_DIR)

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set model to eval mode and move to device
    model.eval()
    model = model.to(device)

    image_list = glob.glob(os.path.join(cfg.SCENE_DIR, "images", "*"))
    image_list = sorted(image_list)
    images = load_and_preprocess_images(image_list)
    images = images[None].to(device, non_blocking=True)  # Use non_blocking for potential speed improvement

    # force to use the shape of 336x518
    
    # List of image counts to test
    # test_image_counts = [1, 2, 4, 8, 10, 20, 50, 100, 200, 400, 800]
    test_image_counts = [2, 4]
    
    # Create a file to save results
    results_file = "benchmark_results.txt"
    with open(results_file, "w") as f:
        f.write("Benchmark Results for Different Image Counts\n")
        f.write("==========================================\n\n")
    
    # Run benchmark for each image count
    for TEST_NUM_IMAGES in test_image_counts:
        print(f"\n\n===== Testing with {TEST_NUM_IMAGES} images =====\n")
        
        # Prepare images for this test
        test_images = images.clone()
        
        if test_images.shape[1] > TEST_NUM_IMAGES:
            test_images = test_images[:, :TEST_NUM_IMAGES, :, :, :]
        else:
            # Repeat images to reach TEST_NUM_IMAGES
            num_available = test_images.shape[1]
            repeats_needed = (TEST_NUM_IMAGES + num_available - 1) // num_available  # Ceiling division
            test_images = test_images.repeat(1, repeats_needed, 1, 1, 1)  # Repeat along dimension 1 (images)
            test_images = test_images[:, :TEST_NUM_IMAGES, :, :, :]  # Trim to exact TEST_NUM_IMAGES
            print(f"Repeated {num_available} images {repeats_needed} times to reach {TEST_NUM_IMAGES} images")

        test_images = test_images[:, :, :, :336, :518]
        print("images.shape", test_images.shape)

        batch = {"images": test_images}

        # Memory and runtime measurement
        num_runs = 10
        runtimes = []
        peak_memories = []

        # Warmup run with torch.cuda.amp.autocast for mixed precision
        # for _ in range(10):
        # with torch.no_grad():
        #     with torch.cuda.amp.autocast(dtype=torch.float16):
        #         _ = model(batch, aggregator_only=True)
                    
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # Clear cache after warmup

        # Preload and pin memory for faster transfers
        test_images = test_images.to(device, non_blocking=True)

        # Main benchmark loop
        for i in range(num_runs):
            # Reset memory stats and clear cache
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            # import pdb; pdb.set_trace()

            # Record memory before model call
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB


            # Create CUDA events for precise timing
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            # Record start time
            start_time.record()
            
            # Run inference with optimized settings
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16): #cache_enabled=True):
                # _ = model(batch, aggregator_only=True)
                _ = model(batch, aggregator_only=False)

            # Record end time
            end_time.record()
                
            # Record peak memory specifically for the model call
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
            memory_used_gb = peak_memory_gb - memory_before

            # Wait for GPU to finish work
            torch.cuda.synchronize()

            # Calculate metrics
            runtime_ms = start_time.elapsed_time(end_time)
            runtime_sec = runtime_ms / 1000  # Convert ms to seconds

            runtimes.append(runtime_ms)
            peak_memories.append(memory_used_gb)  # Store the memory used by model call only

            print(f"Run {i+1}/{num_runs}: {runtime_ms:.2f} ms ({runtime_sec:.4f} s), Memory used: {memory_used_gb:.4f} GB")

        # Calculate and print averages
        avg_runtime = sum(runtimes) / num_runs
        avg_runtime_sec = avg_runtime / 1000  # Convert to seconds
        avg_peak_memory = sum(peak_memories) / num_runs

        # Print results
        result_summary = f"\n===== Results for {TEST_NUM_IMAGES} images =====\n"
        result_summary += f"Average inference runtime: {avg_runtime:.2f} ms ({avg_runtime_sec:.4f} s)\n"
        result_summary += f"Average memory used by model inference: {avg_peak_memory:.4f} GB\n"
        result_summary += f"Min runtime: {min(runtimes):.2f} ms ({min(runtimes)/1000:.4f} s), Max runtime: {max(runtimes):.2f} ms ({max(runtimes)/1000:.4f} s)\n"
        result_summary += f"Min memory used: {min(peak_memories):.4f} GB, Max memory used: {max(peak_memories):.4f} GB\n"
        
        print(result_summary)
        
        # Save results to file
        with open(results_file, "a") as f:
            f.write(result_summary)
            f.write("\n" + "-"*50 + "\n")
        
        # Clean up
        torch.cuda.empty_cache()
        
        # Give GPU some time to cool down between tests
        torch.cuda.synchronize()
    
    # Return the last result for compatibility
    return None



def load_and_preprocess_images(image_path_list):
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # 1. load images as RGB
    # 2. resize images to (518, X, 3), where X is the resized width and X should be divisible by 14
    # 3. normalize images to (0, 1)
    # 4. concatenate images to (N, 3, 518, X), where N is the number of images
    images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    # First process all images and collect their shapes
    for image_path in image_path_list:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        new_width = 518

        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518

        if new_height > 518:
            start_y = (new_height - 518) // 2
            img = img[:, start_y:start_y + 518, :]

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='constant',
                    value=1.0
                )
            padded_images.append(img)
        images = padded_images


    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


if __name__ == "__main__":
    print("Loading model")

    # Set PyTorch performance options
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
    torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for cuDNN
    
    # Enable high precision for float32 matmul
    # torch.set_float32_matmul_precision('high')
    
    # Optional: Set environment variables for performance
    os.environ['PYTORCH_JIT'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
    
    # Set optimal thread settings
    torch.set_num_threads(32)  # Adjust based on your CPU
    
    # Disable dynamo entirely to avoid item() errors
    if hasattr(torch, '_dynamo'):
        torch._dynamo.config.disable = True
    
    cfg_file = "config/base.yaml"
    cfg = OmegaConf.load(cfg_file)

    cfg.AGGREGATOR.use_checkpoint = False
    cfg.CameraHead = None
    cfg.DepthHead = None
    cfg.MatchHead = None
    
    # Instantiate model
    vggt_model = instantiate(cfg, _recursive_=False)

    if True:
        # _CKPT_PATH = "/home/jianyuan/tmp_data/ckpts/r518_t10_v0_0-t5rkf460.pt"
        _CKPT_PATH = "/home/jianyuan/cvpr2025_ckpts/r518_t7_cmh_v7_0-d4w770q_model.pt"

        # /home/jianyuan/cvpr2025_ckpts
        pretrain_model = torch.load(_CKPT_PATH)

        if "model" in pretrain_model:
            model_dict = pretrain_model["model"]
            vggt_model.load_state_dict(model_dict, strict=False)
        else:
            vggt_model.load_state_dict(pretrain_model, strict=False)

        # Free memory
        del pretrain_model
        torch.cuda.empty_cache()
    
    # # Apply torch.compile with optimized settings
    if hasattr(torch, 'compile'):
        print("Using torch.compile for acceleration")
        # Configure dynamo to suppress errors and fall back to eager mode when needed
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        
        vggt_model = torch.compile(
            vggt_model, 
            mode="reduce-overhead",  # Changed from max-autotune to reduce-overhead for better compatibility
            fullgraph=True,         # Changed to False to allow partial graph compilation
            dynamic=False            # Set to True if input sizes vary
        )
        print("compile done")
    
    
    # Optional: Pin memory for faster CPU->GPU transfers
    torch.cuda.empty_cache()
    torch.cuda.memory.empty_cache()
    
    # Run benchmark
    y_hat = demo_fn(cfg, vggt_model)


