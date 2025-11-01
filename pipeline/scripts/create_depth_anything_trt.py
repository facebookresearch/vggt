#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

def parse_shape(s: str):
    # Accept "1x3x518x518" or "1,3,518,518"
    parts = s.replace("x", ",").split(",")
    return tuple(int(p) for p in parts)

def main():
    ap = argparse.ArgumentParser(description="Build a TensorRT engine from ONNX (via Python TRT).")
    ap.add_argument("--onnx", required=True, help="Path to ONNX model")
    ap.add_argument("--engine", required=True, help="Path to save .engine plan")
    ap.add_argument("--fp16", action="store_true", help="Enable FP16")
    ap.add_argument("--workspace-gb", type=float, default=4.0, help="Workspace size in GB (default 4)")
    ap.add_argument("--input-name", type=str, default=None, help="Override ONNX input tensor name")
    ap.add_argument("--shape", type=str, default=None, help="Static shape, e.g. 1x3x518x518")
    ap.add_argument("--min-shape", type=str, default=None, help="Min shape for dynamic input")
    ap.add_argument("--opt-shape", type=str, default=None, help="Opt shape for dynamic input")
    ap.add_argument("--max-shape", type=str, default=None, help="Max shape for dynamic input")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    if not onnx_path.is_file():
        print(f"[ERROR] ONNX not found: {onnx_path}", file=sys.stderr)
        sys.exit(1)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(logger)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        print("[ERROR] ONNX parse failed:", file=sys.stderr)
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        sys.exit(1)

    # Determine input
    if network.num_inputs < 1:
        print("[ERROR] No inputs detected in network.", file=sys.stderr)
        sys.exit(1)
    input_tensor = None
    if args.input_name:
        # Find by name
        for i in range(network.num_inputs):
            if network.get_input(i).name == args.input_name:
                input_tensor = network.get_input(i)
                break
        if input_tensor is None:
            print(f"[ERROR] Input '{args.input_name}' not found. Available:",
                  [network.get_input(i).name for i in range(network.num_inputs)], file=sys.stderr)
            sys.exit(1)
    else:
        input_tensor = network.get_input(0)

    # Build config
    config = builder.create_builder_config()
    # Workspace
    workspace_bytes = int(args.workspace_gb * (1024**3))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes  # TRT < 10 fallback

    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Shapes
    has_dynamic = any(d == -1 for d in input_tensor.shape)
    if args.shape:
        shp = parse_shape(args.shape)
        if has_dynamic:
            profile = builder.create_optimization_profile()
            name = input_tensor.name
            profile.set_shape(name, shp, shp, shp)
            config.add_optimization_profile(profile)
        else:
            network.get_input(0).shape = shp
    else:
        # Optional dynamic profile via min/opt/max
        if args.min_shape and args.opt_shape and args.max_shape:
            min_s = parse_shape(args.min_shape)
            opt_s = parse_shape(args.opt_shape)
            max_s = parse_shape(args.max_shape)
            profile = builder.create_optimization_profile()
            name = input_tensor.name
            profile.set_shape(name, min_s, opt_s, max_s)
            config.add_optimization_profile(profile)
        elif has_dynamic:
            print("[ERROR] Model has dynamic input shape; provide --shape or --min/opt/max.", file=sys.stderr)
            print(f"Input '{input_tensor.name}' current shape: {tuple(input_tensor.shape)}", file=sys.stderr)
            sys.exit(1)

    # Build and save
    serialized = None
    try:
        serialized = builder.build_serialized_network(network, config)
    except AttributeError:
        # TRT < 10
        engine = builder.build_engine(network, config)
        if engine is None:
            print("[ERROR] Engine build failed.", file=sys.stderr)
            sys.exit(1)
        serialized = engine.serialize()

    if serialized is None:
        print("[ERROR] Engine serialization failed.", file=sys.stderr)
        sys.exit(1)

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[OK] Wrote engine: {engine_path}")

if __name__ == "__main__":
    sys.exit(main())