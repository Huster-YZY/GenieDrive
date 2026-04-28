#!/usr/bin/env python3
import os
import pickle
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from tqdm import tqdm
from pyquaternion import Quaternion

from gaussian_renderer import apply_depth_colormap, apply_semantic_colormap, render


cams = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]


def load_occ_gt(occ_path: str):
    layout = np.load(open(occ_path, "rb"), encoding="bytes", allow_pickle=True)
    layout = layout["semantics"]
    return layout


def create_full_center_coords(shape, x_range, y_range, z_range):
    x = torch.linspace(x_range[0], x_range[1], shape[0]).view(-1, 1, 1).expand(shape)
    y = torch.linspace(y_range[0], y_range[1], shape[1]).view(1, -1, 1).expand(shape)
    z = torch.linspace(z_range[0], z_range[1], shape[2]).view(1, 1, -1).expand(shape)
    center_coords = torch.stack((x, y, z), dim=-1)
    return center_coords


def build_static_tensors(device, shape=(200, 200, 16)):
    """
    These tensors are independent of each sample, so they should be built only once
    for each process/GPU instead of being rebuilt for every item.
    """
    xyz = (
        create_full_center_coords(
            shape=shape,
            x_range=(-40.0, 40.0),
            y_range=(-40.0, 40.0),
            z_range=(-1.0, 5.4),
        )
        .view(-1, 3)
        .to(device)
        .float()
    )

    rot = torch.zeros((xyz.shape[0], 4), device=device).float()
    rot[:, 0] = 1.0

    scale = torch.ones((xyz.shape[0], 3), device=device).float() * 0.2

    return xyz, rot, scale


@torch.inference_mode()
def render_occ_semantic_map(
    item_data,
    base_path,
    occ_base_path,
    xyz,
    rot,
    scale,
    device,
    is_vis=False,
    skip_existing=False,
):
    sample_token = item_data["token"]

    save_dir = os.path.join(base_path, sample_token)
    os.makedirs(save_dir, exist_ok=True)

    sem_data_all_path = os.path.join(save_dir, "semantic.npz")
    depth_data_all_path = os.path.join(save_dir, "depth_data.npz")

    if skip_existing and os.path.exists(sem_data_all_path) and os.path.exists(depth_data_all_path):
        return True

    occ_path = os.path.join(occ_base_path, sample_token, "labels.npz")

    if not os.path.exists(occ_path):
        print(f"[Warning] Cannot find {occ_path}")
        return False

    occ_label = load_occ_gt(occ_path=occ_path)

    # Original label remapping logic.
    empty_mask = occ_label == 17
    occ_label[occ_label == 0] = 17
    occ_label[empty_mask] = 0

    # @Optional: object removing
    # remove_mask = occ_label == 4 # 4: car, 3: bus
    # occ_label[remove_mask] = 0

    # @Optional: object insertion
    # bus_ref_path = 'data/gts/scene-0014/2da9ed7919024a0ca4db7a801e70d82a/labels.npz'
    # bus = np.load(bus_ref_path)['semantics']
    # bus_mask = bus == 3
    # occ_label[bus_mask] = 3

    image_shape = (450, 800)

    semantics_np = occ_label.astype(np.float32)
    semantics_tensor = torch.from_numpy(semantics_np).to(device).float()

    # (200, 200, 16) -> (640000, 1)
    semantics_gt = semantics_tensor.view(-1, 1)
    occ_mask = semantics_gt[:, 0] != 0

    # (640000, 1) -> (1, 640000)
    semantics_gt = semantics_gt.permute(1, 0)

    opacity = (semantics_gt.clone() != 0).float()
    opacity = opacity.permute(1, 0).to(device)

    # One-hot semantic feature: (20, N)
    semantics_feat = torch.zeros((20, semantics_gt.shape[1]), device=device).float()
    for i in range(20):
        semantics_feat[i] = semantics_gt == i

    color = torch.zeros((3, semantics_gt.shape[1]), device=device).float()

    rgb = color.permute(1, 0).float()
    feat = semantics_feat.permute(1, 0).float()

    camera_semantic = []
    camera_depth = []

    for cam in cams:
        cam_info = item_data["cams"][cam]

        camera_intrinsic = np.eye(3, dtype=np.float32)
        camera_intrinsic[:3, :3] = np.array(cam_info["cam_intrinsic"], dtype=np.float32)
        camera_intrinsic = torch.from_numpy(camera_intrinsic).to(device).float()

        c2e = Quaternion(cam_info["sensor2ego_rotation"]).transformation_matrix
        c2e[:3, 3] = np.array(cam_info["sensor2ego_translation"])
        camera_extrinsic = torch.from_numpy(c2e).to(device).float()

        # Resize intrinsic for image_shape = (450, 800)
        camera_intrinsic[0][0] = camera_intrinsic[0][0] / 2
        camera_intrinsic[1][1] = camera_intrinsic[1][1] / 2
        camera_intrinsic[0][2] = camera_intrinsic[0][2] / 2
        camera_intrinsic[1][2] = camera_intrinsic[1][2] / 2

        render_pkg = render(
            camera_extrinsic,
            camera_intrinsic,
            image_shape,
            xyz[occ_mask],
            rgb[occ_mask],
            feat[occ_mask],
            rot[occ_mask],
            scale[occ_mask],
            opacity[occ_mask],
            bg_color=[0, 0, 0],
        )

        render_semantic = render_pkg["render_feat"]
        render_depth = render_pkg["render_depth"]

        if is_vis:
            sem_color_dir = os.path.join(save_dir, "semantic_color")
            depth_color_dir = os.path.join(save_dir, "depth_color")
            os.makedirs(sem_color_dir, exist_ok=True)
            os.makedirs(depth_color_dir, exist_ok=True)

            sem_save_path = os.path.join(sem_color_dir, cam + ".png")
            sem_color = (
                apply_semantic_colormap(render_semantic)
                .cpu()
                .permute(1, 2, 0)
                .detach()
                .numpy()
                * 255
            )
            cv2.imwrite(sem_save_path, sem_color)

            depth_save_path = os.path.join(depth_color_dir, cam + ".png")
            depth_vis = torch.clamp(render_depth, min=0.1, max=40.0)
            depth_color = (
                apply_depth_colormap(depth_vis)
                .cpu()
                .permute(1, 2, 0)
                .detach()
                .numpy()
                * 255
            )
            cv2.imwrite(depth_save_path, depth_color)

        semantic = torch.max(render_semantic, dim=0)[1].squeeze().cpu().numpy().astype(np.int8)
        camera_semantic.append(semantic)

        depth_data = render_depth[0].detach().cpu().numpy()
        camera_depth.append(depth_data)

    np.savez(sem_data_all_path, np.array(camera_semantic))
    np.savez(depth_data_all_path, np.array(camera_depth))

    return True


def get_dist_info_from_env():
    """
    Support both:
    1. Single process:
       python render_occ_parallel.py --eval_split train

    2. torchrun multi-GPU:
       torchrun --nproc_per_node=4 render_occ_parallel.py --eval_split train
    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, local_rank, world_size


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data")
    parser.add_argument("--render_path", type=str, default="data/gt_occ_conditions")
    parser.add_argument("--eval_split", type=str, required=True)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")

    # Optional manual distributed arguments.
    # If using torchrun, these are automatically read from environment variables.
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)

    args = parser.parse_args()

    env_rank, env_local_rank, env_world_size = get_dist_info_from_env()

    rank = env_rank if args.rank is None else args.rank
    local_rank = env_local_rank if args.local_rank is None else args.local_rank
    world_size = env_world_size if args.world_size is None else args.world_size

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    print(
        f"[Info] rank={rank}, local_rank={local_rank}, world_size={world_size}, device={device}",
        flush=True,
    )

    eval_mode = args.eval_split

    meta_pkl_path = os.path.join(
        args.dataset_path,
        f"nuscenes_infos_{eval_mode}_temporal_v3_scene.pkl",
    )
    render_base_path = os.path.join(args.render_path, eval_mode)
    occ_base_path = os.path.join(args.dataset_path, "gts")

    if not os.path.exists(meta_pkl_path):
        raise FileNotFoundError(f"Cannot find meta pkl: {meta_pkl_path}")

    os.makedirs(render_base_path, exist_ok=True)

    print(f"[Info] Loading meta file: {meta_pkl_path}", flush=True)
    with open(meta_pkl_path, "rb") as f:
        data = pickle.load(f)

    items_data = data["infos"]

    all_index_list = list(items_data.keys())
    sub_index_list = all_index_list[rank::world_size]

    print(
        f"[Info] Total scenes: {len(all_index_list)}, "
        f"rank {rank} will process {len(sub_index_list)} scenes.",
        flush=True,
    )

    xyz, rot, scale = build_static_tensors(device)

    success_path = os.path.join(render_base_path, f"success_list_rank{rank}.txt")
    fail_path = os.path.join(render_base_path, f"fail_list_rank{rank}.txt")

    for index in tqdm(sub_index_list, desc=f"rank {rank}", dynamic_ncols=True):
        scene = items_data[index]

        scene_success = True

        for item in scene:
            ok = render_occ_semantic_map(
                item_data=item,
                base_path=render_base_path,
                occ_base_path=os.path.join(occ_base_path, index),
                xyz=xyz,
                rot=rot,
                scale=scale,
                device=device,
                is_vis=args.vis,
                skip_existing=args.skip_existing,
            )

            if not ok:
                scene_success = False

        if scene_success:
            with open(success_path, "a") as f:
                f.write(str(index) + "\n")
        else:
            with open(fail_path, "a") as f:
                f.write(str(index) + "\n")

    print(f"[Info] rank {rank} finished.", flush=True)


if __name__ == "__main__":
    main()