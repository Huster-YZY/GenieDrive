import os
import json
import pickle
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

from process_utils import (
    output_video,
    apply_semantic_colormap,
    cam_names,
    cam_dict,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--occ_render_folder", type=str, default="gt_occ_conditions")
    parser.add_argument("--save_folder", type=str, default="videos_train")
    parser.add_argument("--dataset_path", type=str, default="data")
    parser.add_argument("--eval_mode", type=str, default="train")

    # 128 很容易造成 IO 拥塞，建议先试 8 / 16 / 32
    parser.add_argument("--max_workers", type=int, default=16)

    # semantic frame repeat: 2Hz semantic infos -> 12Hz condition video
    parser.add_argument("--semantic_repeat", type=int, default=6)
    parser.add_argument("--fps", type=int, default=12)

    # 用于测试时只处理前 N 个 scene
    parser.add_argument("--num_scenes", type=int, default=None)

    # 如果输出视频已经存在，则跳过
    parser.add_argument("--skip_existing", action="store_true")

    return parser.parse_args()


def convert_nuscenes_path(raw_path, dataset_path):
    """
    Convert paths stored in cfg to local dataset paths.
    """
    path = raw_path.replace("../data/nuscenes/samples/", dataset_path + "/imgs/")
    path = path.replace("../data/nuscenes/sweeps/", dataset_path + "/sweeps/")
    return path


def build_rgb_path_list(cfg, dataset_path, num_scenes=None):
    """
    Build RGB image path lists for each scene and camera.

    Output:
        rgb_path_list[scene_idx][cam_name] = list of image paths
    """
    infos = cfg["infos"]
    scene_tokens = cfg["scene_tokens"]

    if num_scenes is not None:
        scene_tokens = scene_tokens[:num_scenes]

    # token -> info，避免原代码中 index 跳转可能带来的问题
    info_by_token = {info["token"]: info for info in infos}

    rgb_path_list = []

    for cfg_scene in tqdm(scene_tokens, desc="Building RGB path list", dynamic_ncols=True):
        scene_dict = {cam: [] for cam in cam_names}

        for token in cfg_scene:
            if token not in info_by_token:
                continue

            info = info_by_token[token]

            for cam_name in cam_names:
                raw_path = info["cams"][cam_name]["data_path"]
                img_path = convert_nuscenes_path(raw_path, dataset_path)

                if not os.path.exists(img_path):
                    raise FileNotFoundError(
                        f"Image path does not exist: {img_path}\nOriginal path: {raw_path}"
                    )

                scene_dict[cam_name].append(img_path)

        rgb_path_list.append(scene_dict)

    return rgb_path_list


def save_one_rgb_video(frame_list, save_path, skip_existing=False):
    """
    Save one RGB video using the existing output_video function.
    """
    if skip_existing and os.path.exists(save_path):
        return save_path

    if len(frame_list) == 0:
        raise ValueError(f"Empty frame list for {save_path}")

    output_video(frame_list, save_path)
    return save_path


def save_rgb_videos(rgb_path_list, save_folder, max_workers=16, skip_existing=False):
    """
    Save RGB videos in parallel.
    """
    tasks = []

    for scene_idx, scene_cams_list in enumerate(rgb_path_list):
        for cam_name in cam_names:
            save_path = os.path.join(save_folder, f"RGB_{scene_idx:06d}_{cam_name}.mp4")
            frame_list = scene_cams_list[cam_name]
            tasks.append((frame_list, save_path))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                save_one_rgb_video,
                frame_list,
                save_path,
                skip_existing,
            )
            for frame_list, save_path in tasks
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Saving RGB videos",
            dynamic_ncols=True,
        ):
            future.result()


def colorize_one_semantic_view(sem_mv, cam_name):
    """
    Convert one camera semantic map to uint8 color image.

    sem_mv shape is expected to be:
        [num_cameras, H, W]
    """
    sem = sem_mv[cam_dict[cam_name]][None]

    colored_sem = apply_semantic_colormap(sem)
    colored_sem = colored_sem.permute(1, 2, 0).cpu().numpy() * 255
    colored_sem = colored_sem.astype(np.uint8)

    # OpenCV VideoWriter prefers contiguous uint8 array
    colored_sem = np.ascontiguousarray(colored_sem)

    return colored_sem


def write_repeated_frame(writer, frame, repeat):
    """
    Write the same frame multiple times.
    """
    for _ in range(repeat):
        writer.write(frame)


def semantic_outputs_exist(save_folder, scene_idx):
    """
    Check whether all six semantic videos for a scene already exist.
    """
    for cam_name in cam_names:
        path = os.path.join(save_folder, f"SEMANTIC_{scene_idx:06d}_{cam_name}.mp4")
        if not os.path.exists(path):
            return False
    return True


def save_one_scene_semantic_videos(
    scene_idx,
    cfg_scene,
    dataset_path,
    occ_render_folder,
    eval_mode,
    save_folder,
    repeat=6,
    fps=12,
    skip_existing=False,
):
    """
    Stream semantic frames directly into videos for one scene.

    This is the key optimization:
    - It does not store all semantic frames in sem_path_list.
    - It reads one semantic.npz at a time.
    - It writes frames directly to six VideoWriter objects.
    """
    if skip_existing and semantic_outputs_exist(save_folder, scene_idx):
        return scene_idx

    # 2Hz semantic tokens
    tokens = cfg_scene[::6]

    if len(tokens) == 0:
        raise ValueError(f"No semantic tokens found for scene {scene_idx}")

    writers = {cam_name: None for cam_name in cam_names}

    try:
        for token in tokens:
            sem_path = os.path.join(
                dataset_path,
                occ_render_folder,
                eval_mode,
                token,
                "semantic.npz",
            )

            if not os.path.exists(sem_path):
                raise FileNotFoundError(f"Semantic path does not exist: {sem_path}")

            sem_mv = np.load(sem_path)["arr_0"]

            for cam_name in cam_names:
                frame = colorize_one_semantic_view(sem_mv, cam_name)

                if writers[cam_name] is None:
                    height, width = frame.shape[:2]

                    save_path = os.path.join(
                        save_folder,
                        f"SEMANTIC_{scene_idx:06d}_{cam_name}.mp4",
                    )

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open VideoWriter: {save_path}")

                    writers[cam_name] = writer

                write_repeated_frame(writers[cam_name], frame, repeat)

    finally:
        for writer in writers.values():
            if writer is not None:
                writer.release()

    return scene_idx


def save_semantic_videos(
    scene_tokens,
    dataset_path,
    occ_render_folder,
    eval_mode,
    save_folder,
    max_workers=16,
    repeat=6,
    fps=12,
    skip_existing=False,
):
    """
    Save semantic videos scene by scene in parallel.
    """
    tasks = list(enumerate(scene_tokens))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                save_one_scene_semantic_videos,
                scene_idx,
                cfg_scene,
                dataset_path,
                occ_render_folder,
                eval_mode,
                save_folder,
                repeat,
                fps,
                skip_existing,
            )
            for scene_idx, cfg_scene in tasks
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Saving semantic videos",
            dynamic_ncols=True,
        ):
            future.result()


def save_meta_json(save_folder, num_scenes):
    """
    Save meta.json according to the actual number of generated scenes.
    """
    meta = []

    for i in range(num_scenes):
        meta_dict = {"text": "", "type": "video"}

        for cam_name in cam_names:
            meta_dict.update(
                {
                    f"file_path_{cam_name}": f"RGB_{i:06d}_{cam_name}.mp4",
                    f"control_file_path_{cam_name}": f"SEMANTIC_{i:06d}_{cam_name}.mp4",
                }
            )

        meta.append(meta_dict)

    meta_path = os.path.join(save_folder, "meta.json")

    with open(meta_path, "w") as json_file:
        json.dump(meta, json_file, indent=2)

    print(f"Saved meta.json to {meta_path}")


def main():
    args = parse_args()

    # 避免 OpenCV 内部再开很多线程，和 ThreadPoolExecutor 互相抢资源
    cv2.setNumThreads(1)

    dataset_path = args.dataset_path
    occ_render_folder = args.occ_render_folder
    save_folder = os.path.join(dataset_path, args.save_folder)
    eval_mode = args.eval_mode
    max_workers = args.max_workers

    os.makedirs(save_folder, exist_ok=True)

    cfg_path = os.path.join(
        dataset_path,
        f"nuscenes_interp_12Hz_infos_{eval_mode}.pkl",
    )

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Cannot find cfg file: {cfg_path}")

    print(f"Loading cfg from: {cfg_path}")

    with open(cfg_path, "rb") as f:
        cfg = pickle.load(f)

    scene_tokens = cfg["scene_tokens"]

    if args.num_scenes is not None:
        scene_tokens = scene_tokens[: args.num_scenes]

    num_scenes = len(scene_tokens)

    print(f"Number of scenes to process: {num_scenes}")
    print(f"Save folder: {save_folder}")
    print(f"Max workers: {max_workers}")
    print(f"Skip existing: {args.skip_existing}")

    # 1. Build RGB path list
    rgb_path_list = build_rgb_path_list(
        cfg=cfg,
        dataset_path=dataset_path,
        num_scenes=args.num_scenes,
    )

    # 2. Save RGB videos
    save_rgb_videos(
        rgb_path_list=rgb_path_list,
        save_folder=save_folder,
        max_workers=max_workers,
        skip_existing=args.skip_existing,
    )

    # 3. Save semantic videos directly without storing sem_path_list
    save_semantic_videos(
        scene_tokens=scene_tokens,
        dataset_path=dataset_path,
        occ_render_folder=occ_render_folder,
        eval_mode=eval_mode,
        save_folder=save_folder,
        max_workers=max_workers,
        repeat=args.semantic_repeat,
        fps=args.fps,
        skip_existing=args.skip_existing,
    )

    # 4. Save meta.json
    save_meta_json(
        save_folder=save_folder,
        num_scenes=num_scenes,
    )

    print("done")


if __name__ == "__main__":
    main()