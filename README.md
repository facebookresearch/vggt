<div align="center">
<h1>MuM: Feed-forward Reconstruction Evaluation</h1>
</div>

## Overview
This is the basis for our experiments on feed-forward reconstruction in the MuM paper. It is built around the idea of changing the backbone in VGGT from DINO to different encoders. This repo supports MuM, CroCo v2, and DINOv3.

## Installations
The environment requirements are the same as in VGGT, so doing the same as in VGGT, i.e.
```bash
pip install -r requirements.txt
```
should work fine. 

## Data
You can see the datasets we trained on in `training/data`. We provide the annotation files [HERE](https://github.com/davnords/vggt/releases/download/annotations/annotations.zip) for you to download. I recommend you simply start with downloading MegaDepth by following the directions in the [DKM](https://github.com/parskatt/dkm) and [RoMa](https://github.com/parskatt/roma) repos.

## Training
I launched the training by running the sbatch script `training/vggt.sh`. However, the most common usage would be to run:
```bash
cd training
srun torchrun \
  --nproc_per_node=8 \
  --nnodes=4 \
  --rdzv_backend=c10d \
  launch.py --config "dinov3"
```
Depending on how many GPUs you have available. Look at the `training/config` files to see all the default choices we made and also how to change model (by selecting the patch embed). Feel free to train by altering the configs to fit the size of your training run needs.

## Evaluation
For evaluation, we mainly ran MegaDepth and Re10k. The following scripts were what we ran:
```bash
cd evaluation
python test_relpose.py --data_dir data/megadepth --anno_dir annotations/megadepth/test.jgz --model_path ../training/logs/mum_exp001/ckpts/checkpoint.pt --encoder mum 

python test_relpose.py --data_dir data/re10k/ --anno_dir annotations/re10k/test.jgz --model_path ../training/logs/mum_exp004/ckpts/checkpoint.pt --encoder mum
```

## Disclaimer
I have not thoroughly tested this code outside of my own setup, so there might be some small edits you need to make to make it work for you (e.g. changing the absolute filepaths in the configs and designing your own sbatch script). If you want to add your own model, you can see in `vggt/models/aggregator_small.py` how to do this.

## Checkpoints
I have put them out as a GitHub release. They can be found [here](https://github.com/davnords/vggt/releases/tag/weights).
