import argparse
import statistics

import torch

from vggt.models.aggregator import Aggregator


MIB = 2**20


def run_once(aggregator, images, chunk_size, autocast_enabled):
    aggregator.set_inference_token_chunk_size(chunk_size)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base_allocated = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    with (
        torch.inference_mode(),
        torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
        ),
    ):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output, _ = aggregator(images)
        end.record()
        end.synchronize()

    elapsed_ms = start.elapsed_time(end)
    peak_delta_mb = (torch.cuda.max_memory_allocated() - base_allocated) / MIB
    reserved_delta_mb = (torch.cuda.max_memory_reserved() - base_reserved) / MIB
    cached_output_mb = (
        sum(
            tensor.numel() * tensor.element_size()
            for tensor in output
            if tensor is not None
        )
        / MIB
    )
    del output
    torch.cuda.synchronize()
    return elapsed_ms, peak_delta_mb, reserved_delta_mb, cached_output_mb


def measure(aggregator, images, chunk_size, repeats, autocast_enabled):
    times = []
    peaks = []
    reserved_peaks = []
    cached_output_mb = None
    for _ in range(repeats):
        elapsed_ms, peak_mb, reserved_mb, cached_output_mb = run_once(
            aggregator, images, chunk_size, autocast_enabled
        )
        times.append(elapsed_ms)
        peaks.append(peak_mb)
        reserved_peaks.append(reserved_mb)
    return statistics.median(times), max(peaks), max(reserved_peaks), cached_output_mb


def build_aggregator(patch_embed):
    return Aggregator(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        patch_embed=patch_embed,
        qk_norm=True,
        rope_freq=100,
        cached_layer_indices=(4, 11, 17, 23),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, nargs="+", default=[40, 80, 120])
    parser.add_argument("--chunk", type=int, default=16384)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument(
        "--patch-embed",
        default="dinov2_vitl14_reg",
        choices=["conv", "dinov2_vitl14_reg"],
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    parameter_dtype = torch.float32 if args.autocast else torch.bfloat16
    input_dtype = torch.float32 if args.autocast else torch.bfloat16
    with torch.device(device):
        aggregator = build_aggregator(args.patch_embed)
    aggregator = aggregator.to(device=device, dtype=parameter_dtype).eval()

    warmup = torch.rand((1, 1, 3, 518, 518), device=device, dtype=input_dtype)
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.autocast),
    ):
        output = aggregator(warmup)
    del output, warmup
    torch.cuda.synchronize()

    print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
    print(
        f"patch_embed={args.patch_embed} autocast={args.autocast} "
        f"model_mb={torch.cuda.memory_allocated()/MIB:.2f}"
    )
    for frames in args.frames:
        images = torch.rand((1, frames, 3, 518, 518), device=device, dtype=input_dtype)
        baseline = measure(aggregator, images, None, args.repeats, args.autocast)
        streamed = measure(aggregator, images, args.chunk, args.repeats, args.autocast)

        reduction = (1 - streamed[1] / baseline[1]) * 100
        slowdown = streamed[0] / baseline[0]
        print(
            f"frames={frames} chunk={args.chunk} "
            f"baseline_ms={baseline[0]:.3f} stream_ms={streamed[0]:.3f} "
            f"slowdown={slowdown:.4f}x "
            f"baseline_peak_delta_mb={baseline[1]:.2f} "
            f"stream_peak_delta_mb={streamed[1]:.2f} "
            f"peak_reduction={reduction:.2f}% "
            f"baseline_reserved_delta_mb={baseline[2]:.2f} "
            f"stream_reserved_delta_mb={streamed[2]:.2f} "
            f"cached_output_mb={baseline[3]:.2f}"
        )

        del images
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
