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

Then your dictory should look like:
```
occ_gen
├── mmdet3d
├── ckpts/
│   ├── genie_occ.pth
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