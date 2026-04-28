import os
import time
import pickle
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from process_utils import output_video, output_sem_video, apply_semantic_colormap, cam_names, cam_dict

## load Semantic video
rgb_path_list = []
sem_path_list = []
ref_img_list = []

parser = argparse.ArgumentParser()
parser.add_argument("--occ_render_folder", type=str, default="gt_occ_conditions")
parser.add_argument("--save_folder", type=str, default="eval_videos")
parser.add_argument("--dataset_path", type=str, default="data")
parser.add_argument("--eval_mode", type=str, default="val")
parser.add_argument("--selected_scene_index", type=int, default=None)
parser.add_argument("--num_semantic_frames", type=int, default=14)
parser.add_argument("--max_workers", type=int, default=128)

args = parser.parse_args()

dataset_path = args.dataset_path
occ_render_folder = args.occ_render_folder
save_folder = os.path.join(dataset_path, args.save_folder)
eval_mode = args.eval_mode
selected_scene_index = args.selected_scene_index
num_semantic_frames = args.num_semantic_frames
num_rgb_frames = 1 + num_semantic_frames * 6
max_workers = args.max_workers

def worker(func, frame_list, save_path):
    func(frame_list, save_path)

cfg_path = os.path.join(dataset_path, f'nuscenes_interp_12Hz_infos_{eval_mode}.pkl')
with open(cfg_path, 'rb') as f:
    cfg = pickle.load(f)


os.makedirs(save_folder, exist_ok=True)

# # 1. process RGB infos
index = 0
for cfg_scene in tqdm(cfg['scene_tokens']):
    scene_dict = {cam:[] for cam in cam_names}
    ref_dict = {cam:None for cam in cam_names}
    frame_cnt = 0
    while index<len(cfg['infos']) and cfg['infos'][index]['token'] in cfg_scene:
        info = cfg['infos'][index]
        for cam_name in cam_names:
            info_path = info['cams'][cam_name]['data_path']
            info_path = info_path.replace('../data/nuscenes/samples/', dataset_path + '/imgs/')
            info_path = info_path.replace('../data/nuscenes/sweeps/', dataset_path + '/sweeps/')
            assert os.path.exists(info_path), f"Path does not exist: {info_path},\n {info['cams'][cam_name]['data_path']}"
            scene_dict[cam_name].append(info_path)
            if frame_cnt ==0:
                ref_dict[cam_name] = info_path
        frame_cnt += 1
        index += 1
        if frame_cnt == num_rgb_frames:
            break

    index += len(cfg_scene) - frame_cnt
    rgb_path_list.append(scene_dict)
    ref_img_list.append(ref_dict)
    if index == len(cfg['infos']):
        break

#save GT videos
idx = selected_scene_index if selected_scene_index is not None else 0
tasks = []
for scene_cams_list in rgb_path_list:
    for cam_name in cam_names:
        save_path = os.path.join(save_folder, f'RGB_{idx:06d}_{cam_name}.mp4')
        tasks.append((scene_cams_list[cam_name], save_path))
    idx += 1

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(worker, output_video, frame_list, save_path) for frame_list, save_path in tasks]
    for future in tqdm(as_completed(futures), total=len(futures)):
        future.result()

#save ref image
idx = selected_scene_index if selected_scene_index is not None else 0
for ref_imgs in tqdm(ref_img_list):
    for cam_name in cam_names:
        shutil.copy(ref_imgs[cam_name], os.path.join(save_folder, f'REF_{idx:06d}_{cam_name}.jpg'))
    idx = idx + 1

# 2. process semantic maps
def process_one_token(token, dataset_path, occ_render_folder, eval_mode, cam_names, cam_dict):
    sem_path = os.path.join(dataset_path, f"{occ_render_folder}/{eval_mode}", token, "semantic.npz")
    assert os.path.exists(sem_path), f"Semantic path does not exist: {sem_path}"
    sem_mv = np.load(sem_path)["arr_0"]
    result = {}
    for cam_name in cam_names:
        sem = sem_mv[cam_dict[cam_name]][None]
        colored_sem = apply_semantic_colormap(sem).permute(1, 2, 0).cpu().numpy() * 255
        colored_sem = colored_sem.astype(np.uint8)
        result[cam_name] = colored_sem

    return token, result

for scene_index, cfg_scene in tqdm(enumerate(cfg["scene_tokens"]), total=len(cfg["scene_tokens"])):
    if selected_scene_index is not None and scene_index < selected_scene_index:
        continue
    elif selected_scene_index is not None and scene_index > selected_scene_index:
        break

    scene_dict = {cam: [] for cam in cam_names}
    tokens = cfg_scene[::6][:num_semantic_frames]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                lambda token: process_one_token(
                    token,
                    dataset_path,
                    occ_render_folder,
                    eval_mode,
                    cam_names,
                    cam_dict,
                ),
                tokens,
            )
        )

    for token, token_result in results:
        for cam_name in cam_names:
            scene_dict[cam_name].append(token_result[cam_name])

    sem_path_list.append(scene_dict)


#save semantic videos
idx = selected_scene_index if selected_scene_index is not None else 0
tasks = []
for scene_cams_list in sem_path_list:
    for cam_name in cam_names:
        save_path = os.path.join(save_folder, f'SEMANTIC_{idx:06d}_{cam_name}.mp4')
        tasks.append((scene_cams_list[cam_name], save_path))
    idx += 1

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(worker, output_sem_video, frame_list, save_path) for frame_list, save_path in tasks]
    for future in tqdm(as_completed(futures), total=len(futures)):
        future.result()

print("done")