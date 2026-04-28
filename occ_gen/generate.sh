sample_idx=$1

formatted_idx=$(printf "%04d" "$sample_idx")

python tools/generate.py configs/world_model/vae_e2e.py ckpts/genie_occ.pth \
--generate_path generate_output --generate_scene_name scene-${formatted_idx} --generate_frame 12 --task_mode generate

# python tools/generate.py configs/world_model/vae_e2e.py ckpts/genie_occ.pth \
# --generate_path generate_output --generate_scene_name scene-${formatted_idx} --generate_frame 12 --task_mode generate --save_npz