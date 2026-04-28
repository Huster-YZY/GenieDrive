# Occupancy Conditioned Video Generation Setup

## Environment
```bash
cd occ_render
conda create -n geniedrive-video python=3.12 -y
conda activate geniedrive-video
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt

cd occ_rasterizer/diff-gaussian-rasterization
pip install . #Note: CUDA Toolkit is required to build and install this package.
```

## Dataset
Prepare the conditions (ref frame + rendered multi-view semantic maps sequences) for video generation:

Configure soft link (skip this step if you have done):
```bash
ln -s [your Nuscenes path] data
```
### ⚡️Quick Start
If you only want to quickly try the video generation pipeline without processing the full dataset, we also provide several rendered examples that can be used directly:

```bash
huggingface-cli download --resume-download ANIYA673/GenieDrive --include="eval_*" --local-dir data --local-dir-use-symlinks False

```

After downloading the checkpoint as described below, you can directly generate videos using the following commands:
```bash
python examples/wan2.1_fun/driving_video_generation.py --ckpt_path ckpts --sample_condition_path data/eval_insertion --video_length 81

python examples/wan2.1_fun/driving_video_generation.py --ckpt_path ckpts --sample_condition_path data/eval_removing --video_length 81

python examples/wan2.1_fun/driving_video_generation.py --ckpt_path ckpts --sample_condition_path data/eval_videos_roll --video_length 37

python examples/wan2.1_fun/driving_video_generation.py --ckpt_path ckpts --sample_condition_path data/eval_videos_pitch --video_length 37
```

### Process the Dataset for Evaluation

Before processing the evaluation dataset, please make sure that you have rendered the occupancy annotations into semantic maps, as described in [occ_rasterizer/README.md](../occ_rasterizer/README.md).
```bash
python data_processing/process_eval_cond.py \
    --occ_render_folder gt_occ_conditions \
    --dataset_path data \
    --save_folder eval_videos \
    --eval_mode val
```

to generate longer videos:
```bash
python data_processing/process_eval_cond.py \
    --occ_render_folder gt_occ_conditions \
    --dataset_path data \
    --num_semantic_frames 40 \
    --save_folder eval_videos_rollout \
    --eval_mode val
```

## Checkpoint Download
download base weight of Wan2.1
```bash
huggingface-cli download --resume-download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control --local-dir models/Diffusion_Transformer --local-dir-use-symlinks False
```

download our checkpoint:
```bash
huggingface-cli download --resume-download ANIYA673/GenieDrive --include="diffusion_pytorch_model.safetensors" --local-dir ckpts --local-dir-use-symlinks False
```

Then your dictory should look like:

```
.
└── occ_render/
    ├── data/
    ├── models/
    │   └── Diffusion_Transformer/
    └── ckpts/
        ├── config.json
        └── diffusion_pytorch_model.safetensors
```

## Inference

~6s Video Generation:
```bash
torchrun --nproc-per-node 1 examples/wan2.1_fun/predict_v2v_control_ref.py --ckpt_path ckpts --cond_path data/eval_videos
```

~20s Video Generation:
```bash
torchrun --nproc-per-node 1 examples/wan2.1_fun/rollout_long.py --ckpt_path ckpts --cond_path /data4/zhenya/driving_videos/eval_videos_rollout
```

## Training

Processing the dataset for training: (It may take some times.)
```bash
python data_processing/process_training_data.py \
    --occ_render_folder gt_occ_conditions \
    --dataset_path data \
    --save_folder videos_train \
    --eval_mode train \
    --max_workers 16
```

```bash
./train.sh
```