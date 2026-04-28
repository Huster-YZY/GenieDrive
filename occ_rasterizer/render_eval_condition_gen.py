import os
import pickle
from tqdm import tqdm
import cv2
import numpy as np
import torch
from gaussian_renderer import apply_depth_colormap, apply_semantic_colormap, render
from pyquaternion import Quaternion

## 200 occ3d
cams = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]

def output_sem_video(sem_list, output_path, first_ref = False):
    new_frames = []
    for i, array in enumerate(sem_list):
        if first_ref and i==0:
            new_frames.append(array)
            continue
        for _ in range(6):
            new_frames.append(array)
    fps = 12
    height, width, _ = new_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in new_frames:
        # print(type(frame), frame.shape, frame)
        out.write(frame)
    out.release()

def render_occ_semantic_map(item_data, base_path, occ_base_path, is_vis=False, precomputed_occ = None, pre_occ_idx = -1):
    # dict_keys(['lidar_path', 'token', 'prev', 'next', 'can_bus', 'frame_idx', 'sweeps', 'cams', 'scene_token',
    #            'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation',
    #            'timestamp', 'occ_gt_path', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts',
    #            'valid_flag'])

    sample_token = item_data["token"]

    if precomputed_occ is None:
        occ_path = os.path.join(occ_base_path, sample_token, "labels.npz")
        if not os.path.exists(occ_path):
            print(f"can not find {occ_path}")
            return 
        occ_label = load_occ_gt(occ_path=occ_path)
    else:
        occ_label = precomputed_occ
        sample_token = str(pre_occ_idx)

    # occ_label[occ_label == 17] = 0  # 17 -> 0

    empty_mask = occ_label==17
    occ_label[occ_label == 0] = 17
    occ_label[empty_mask] = 0


    semantics = occ_label

    image_shape = (450, 800)

    semantics = torch.from_numpy(semantics.astype(np.float32))  # 200, 200, 16
    xyz = (
        create_full_center_coords(shape=(200, 200, 16), x_range=(-40.0, 40.0), y_range=(-40.0, 40.0), z_range=(-1, 5.4))
        .view(-1, 3)
        .cuda()
        .float()
    )

    # load semantic data ------------------------------------------------------------------------------
    semantics_gt = semantics.view(-1, 1)  # (200, 200, 16) -> (640000, 16)
    occ_mask = semantics_gt[:, 0] != 0
    semantics_gt = semantics_gt.permute(1, 0)

    opacity = (semantics_gt.clone() != 0).float()
    opacity = opacity.permute(1, 0).cuda()

    semantics = torch.zeros((20, semantics_gt.shape[1])).cuda().float()
    color = torch.zeros((3, semantics_gt.shape[1])).cuda()
    for i in range(20):
        semantics[i] = semantics_gt == i

    rgb = color.permute(1, 0).float()
    feat = semantics.permute(1, 0).float()
    rot = torch.zeros((xyz.shape[0], 4)).cuda().float()
    rot[:, 0] = 1
    scale = torch.ones((xyz.shape[0], 3)).cuda().float() * 0.2

    camera_semantic = []
    camera_depth = []
    if not os.path.exists(os.path.join(base_path, sample_token)):
        os.makedirs(os.path.join(base_path, sample_token))
    sem_data_all_path = os.path.join(base_path, sample_token, "semantic.npz")
    depth_data_all_path = os.path.join(base_path, sample_token, "depth_data.npz")

    mv_sem_dict = {}

    for cam in cams:
        cam_info = item_data["cams"][cam]
        camera_intrinsic = np.eye(3).astype(np.float32)
        # print(cam_info)
        camera_intrinsic[:3, :3] = cam_info["cam_intrinsic"]
        camera_intrinsic = torch.from_numpy(camera_intrinsic).cuda().float()

        c2e = Quaternion(cam_info["sensor2ego_rotation"]).transformation_matrix
        c2e[:3, 3] = np.array(cam_info["sensor2ego_translation"])
        c2e = torch.from_numpy(c2e).cuda().float()

        camera_extrinsic = c2e

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

        render_pkg["render_color"]
        render_semantic = render_pkg["render_feat"]
        render_depth = render_pkg["render_depth"]
        render_pkg["render_alpha"]

        if is_vis:
            if not os.path.exists(os.path.join(base_path, sample_token, "semantic_color")):
                os.makedirs(os.path.join(base_path, sample_token, "semantic_color"))
            sem_save_path = os.path.join(base_path, sample_token, "semantic_color", cam + ".png")
            with open(sem_save_path, "wb") as f:
                data = apply_semantic_colormap(render_semantic).cpu().permute(1, 2, 0).detach().numpy() * 255
                f.write(cv2.imencode(".png", data)[1])
            
            # print(cam)
            mv_sem_dict[cam] = data.astype(np.uint8)

            if not os.path.exists(os.path.join(base_path, sample_token, "depth_color")):
                os.makedirs(os.path.join(base_path, sample_token, "depth_color"))
            depth_save_path = os.path.join(base_path, sample_token, "depth_color", cam + ".png")
            with open(depth_save_path, "wb") as f:
                render_depth = torch.clamp(render_depth, min=0.1, max=40.0)
                data = apply_depth_colormap(render_depth).cpu().permute(1, 2, 0).detach().numpy() * 255
                f.write(cv2.imencode(".png", data)[1])

        semantic = torch.max(render_semantic, dim=0)[1].squeeze().cpu().numpy().astype(np.int8)
        camera_semantic.append(semantic)

        depth_data = render_depth[0].detach().cpu().numpy()
        camera_depth.append(depth_data)

    ################################### update object to local ####################################
    np.savez(sem_data_all_path, camera_semantic)
    np.savez(depth_data_all_path, camera_depth)

    # print(f"Rendered {sample_token} to {base_path}{sample_token}")
    return mv_sem_dict


def load_occ_gt(occ_path: str):
    layout = np.load(open(occ_path, "rb"), encoding="bytes", allow_pickle=True)
    # print(layout.keys())
    layout = layout["semantics"]
    return layout


def create_full_center_coords(shape, x_range, y_range, z_range):
    x = torch.linspace(x_range[0], x_range[1], shape[0]).view(-1, 1, 1).expand(shape)
    y = torch.linspace(y_range[0], y_range[1], shape[1]).view(1, -1, 1).expand(shape)
    z = torch.linspace(z_range[0], z_range[1], shape[2]).view(1, 1, -1).expand(shape)

    center_coords = torch.stack((x, y, z), dim=-1)

    return center_coords


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data")
    parser.add_argument("--generated_occ_path", type=str, default="../occ_gen/generate_output/")
    parser.add_argument("--render_path", type=str, default="./gen_occ_conditions/")
    parser.add_argument("--eval_split", type=str, required = True)
    parser.add_argument("--vis", action="store_true")

    args = parser.parse_args()

    # render train data
    eval_mode = args.eval_split
    train_pkl_path = os.path.join(args.dataset_path, f"world-nuscenes_infos_{eval_mode}.pkl")
    render_base_path = os.path.join(args.render_path, eval_mode)

    occ_base_path = os.path.join(args.dataset_path, "gts")
    data = pickle.load(open(train_pkl_path, "rb"))
    items_data = data["infos"]


    #todo: move to argument
    cur_index = 1364
    scene_index = 270
    picked_frame_idx = 15

    scene_name = f'scene-{scene_index:04d}'
    patterns = ['ori', 'sam', 'sam_2']
    item = items_data[cur_index]

    for mode in range(len(patterns)):
        occ_path = os.path.join(args.generated_occ_path, f"{scene_name}/{picked_frame_idx}_{patterns[mode]}.npz")
        occs = np.load(occ_path)['semantics']
        
        sem_frames = {}
        for cam in cams:
            sem_frames[cam] = []
        # render real condition occ
        res_dict = render_occ_semantic_map(
            item, render_base_path, occ_base_path=os.path.join(occ_base_path, scene_name), is_vis=args.vis
        )

        for cam in cams:
            sem_frames[cam].append(res_dict[cam])
        
        for i, occ in enumerate(occs):
            res_dict = render_occ_semantic_map(
                item, render_base_path, occ_base_path=os.path.join(occ_base_path, scene_name), is_vis=args.vis, precomputed_occ=occ, pre_occ_idx=i
            )
            print(f"Rendered {i} frame.")
            for cam in cams:
                sem_frames[cam].append(res_dict[cam])
        
        for cam in cams:
            output_sem_video(sem_frames[cam], os.path.join(render_base_path, f'scene-{scene_index:04d}_' + patterns[mode], f'{cam}.mp4'))


        print(f'scene-{scene_index:06d}-' + patterns[mode] + " done.")
