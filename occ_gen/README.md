# Occupancy World Model Set Up

## Environment
Install Pytorch 1.13 + CUDA 11.6

```setup
conda create --name geniedrive-occ python=3.8
conda activate geniedrive-occ
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

Install mmdet3d (v1.0.0rc4) related packages and build this project
```setup
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
pip install mmengine
pip install -v -e .
```

Install other dependencies
```setup
pip install -r requirements.txt
```

## Checkpoint Download
```bash
huggingface-cli download --resume-download ANIYA673/GenieDrive --include="genie_occ.pth" --local-dir ckpts --local-dir-use-symlinks False
```

To download the pretrained VAE before the end-to-end tuning:
```bash
huggingface-cli download --resume-download ANIYA673/GenieDrive --include="vae_d64_210.pth" --local-dir ckpts --local-dir-use-symlinks False
```
Please refer to [vae/README.md](./vae/README.md) if you want to train the VAE from scratch.

Then your dictory should look like:
```
occ_gen
├── mmdet3d
├── ckpts/
│   ├── genie_occ.pth
│   ├── vae_d64_210.pth
├── data/
│   ├── nuscenes/
│   │   ├── samples/ 
│   │   ├── v1.0-trainval/
│   │   ├── gts/ (Occ3D-nus)
│   │   ├── world-nuscenes_infos_train.pkl
│   │   ├── world-nuscenes_infos_val.pkl
```

## Evaluation
```bash
./test.sh
```

## Training
The training process of the occupancy forecasting model consists of two stages. In the first stage, we freeze the VAE and train only the forecasting module. In the second stage, we further perform end-to-end training.

Please note that, during development, we trained the model iteratively while exploring different configurations, including different ways of injecting the planning signal, such as addition or attention, as well as different supervision losses, such as feature-level L2 loss or feature loss combined with Lovász-Softmax loss. Therefore, it is difficult to provide a single standardized and fully reproducible procedure for Stage 1.

To facilitate reproduction, we provide our trained Stage-1 model as the initialization for Stage 2. The Stage-2 training procedure is listed as follows:

```bash
huggingface-cli download --resume-download ANIYA673/GenieDrive --include="stage2_init_ckpt.pth" --local-dir ckpts --local-dir-use-symlinks False

bash tools/dist_train.sh configs/world_model/vae_e2e.py 8
```