export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"
export DATASET_NAME="data/videos_train"
export DATASET_META_NAME="data/videos_train/meta.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_port="2345" --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_fun/train_control.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=2 \
  --video_sample_n_frames=37 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=10 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="model_training" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --fix_sample_size 256 512\
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --use_deepspeed \
  --train_mode="control_ref" \
  --control_ref_image="first_frame" \
  --add_full_ref_image_in_self_attention \
  --trainable_modules "." \
  --resume_from_checkpoint="latest"