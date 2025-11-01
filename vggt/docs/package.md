# Alternative Installation Methods

This document explains how to install VGGT as a package using different package managers. 

## Prerequisites

Before installing VGGT as a package, you need to install PyTorch and torchvision. We don't list these as dependencies to avoid CUDA version mismatches. Install them first, with an example as:

```bash
# install pytorch 2.3.1 with cuda 12.1
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

## Installation Options

### Install with pip

The simplest way to install VGGT is using pip:

```bash
pip install -e .
```

### Install and run with pixi

[Pixi](https://pixi.sh) is a package management tool for creating reproducible environments.

1. First, [download and install pixi](https://pixi.sh/latest/get_started/)
2. Then run:

```bash
pixi run -e python demo_gradio.py
```

### Install and run with uv

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver.

1. First, [install uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Then run:

```bash
uv run --extra demo demo_gradio.py
```

