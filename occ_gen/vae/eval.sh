torchrun --nproc_per_node=8 eval.py \
  --data_path ../data/nuscenes \
  --cfg_path configs/dataset.yaml \
  --ckpt_path ../ckpts/vae_d64_210.pth