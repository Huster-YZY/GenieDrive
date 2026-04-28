import copy
import os
import pickle

import cv2
import numba as nb
import numpy as np
import open3d
import torch
from gaussian_renderer import apply_depth_colormap, apply_semantic_colormap, render
from pyquaternion import Quaternion

cams = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]


def replace_occ_grid_with_bev(
    input_occ, bevlayout, driva_area_idx=11, bev_replace_idx=[1, 5, 6], occ_replace_new_idx=[17, 18, 19]
):
    # self.classes= ['drivable_area','ped_crossing','walkway','stop_line','carpark_area','road_divider','lane_divider','road_block']
    # occ road [11] drivable area

    # default ped_crossing->18; stop_line->19 (del); roal_divider->20; lane_divider->21
    # default shape: input_occ: [200,200,16]; bevlayout: [18,200,200]

    roal_divider_mask = bevlayout[5, :, :].astype(np.uint8)
    lane_divider_mask = bevlayout[6, :, :].astype(np.uint8)

    roal_divider_mask = cv2.dilate(roal_divider_mask, np.ones((3, 3), np.uint8))
    lane_divider_mask = cv2.dilate(lane_divider_mask, np.ones((3, 3), np.uint8))

    bevlayout[5, :, :] = roal_divider_mask.astype(bool)
    bevlayout[6, :, :] = lane_divider_mask.astype(bool)

    n = len(bev_replace_idx)
    x_max, y_max = input_occ.shape[0], input_occ.shape[1]
    output_occ = input_occ.copy()  # numpy copy() ; tensor clone()
    bev_replace_mask = []
    for i in range(n):
        bev_replace_mask.append(bevlayout[bev_replace_idx[i]] == 1)

    for x in range(x_max):
        for y in range(y_max):
            for i in range(n):
                if bev_replace_mask[i][x, y]:
                    occupancy_data = input_occ[x, y, :]

                    if driva_area_idx in occupancy_data:
                        max_11_index = np.where(occupancy_data == driva_area_idx)
                        output_occ[x, y, max_11_index] = occ_replace_new_idx[i]
    return output_occ


def load_occ_layout(layout_path):
    # load layout data
    layout = np.load(open(layout_path, "rb"), encoding="bytes", allow_pickle=True)
    layout = layout["bev_map"]
    return layout


def render_occ_semantic_map(item_data, base_path, occ_base_path, layout_base_path, is_vis=False):
    # dict_keys(['lidar_path', 'token', 'prev', 'next', 'can_bus', 'frame_idx', 'sweeps', 'cams', 'scene_token',
    #            'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation',
    #            'timestamp', 'occ_gt_path', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts',
    #            'valid_flag'])

    occ_base_path = "gs_render/data/GT_occupancy/dense_voxels_with_semantic/"
    layout_base_path = "nuscenes/12hz_bevlayout_800_800/"
    is_vis = True

    sample_token = item_data["token"]
    lidar_token = os.listdir(os.path.join(occ_base_path, sample_token))[0][:-4]

    occ_path = os.path.join(occ_base_path, sample_token, lidar_token + ".npy")
    occ_label = load_occ_gt(occ_path=occ_path, grid_size=np.array([800, 800, 64]))

    layout_path = os.path.join(layout_base_path, sample_token + ".npz")
    bevlayout = load_occ_layout(layout_path=layout_path)

    semantics = occ_label
    semantics = replace_occ_grid_with_bev(semantics, bevlayout)

    image_shape = (450, 800)

    semantics = torch.from_numpy(semantics.astype(np.float32))  # 200, 200, 16
    xyz = create_full_center_coords(shape=(800, 800, 64)).view(-1, 3).cuda().float()
    l2e = Quaternion(item_data["lidar2ego_rotation"]).transformation_matrix
    l2e[:3, 3] = np.array(item_data["lidar2ego_translation"])
    l2e = torch.from_numpy(l2e).cuda().float()
    xyz = l2e[:3, :3] @ xyz.t() + l2e[:3, 3:4]
    xyz = xyz.t()

    # add for filter floaters
    semantics_gt = semantics.view(-1, 1)  # (512, 512, 40) -> (10485760, 16)
    occ_mask = semantics_gt[:, 0] != 0
    pts = xyz[occ_mask].clone().cpu().numpy()
    colors = np.ones_like(pts)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    pcd, idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    inlier_xyz = torch.from_numpy(np.array(pcd.points))  # (n,3)
    inlier_xyz_lidar = (
        torch.inverse(l2e) @ torch.hstack([inlier_xyz, torch.ones(inlier_xyz.shape[0], 1)]).cuda().float().T
    )
    inlier_xyz = inlier_xyz_lidar[:3, :].T

    pc_range = [-50.0, -50.0, -5, 50.0, 50.0, 3]
    inlier_idx = torch.vstack(
        [
            torch.clamp(800 * (inlier_xyz[:, 0] - pc_range[0]) / (pc_range[3] - pc_range[0]), 0, 800 - 1),
            torch.clamp(800 * (inlier_xyz[:, 1] - pc_range[1]) / (pc_range[4] - pc_range[1]), 0, 800 - 1),
            torch.clamp(64 * (inlier_xyz[:, 2] - pc_range[2]) / (pc_range[5] - pc_range[2]), 0, 64 - 1),
        ]
    ).long()

    filter_mask = torch.zeros((800, 800, 64)).cuda()
    filter_mask[inlier_idx[0, :], inlier_idx[1, :], inlier_idx[2, :]] = 1.0
    semantics[filter_mask.cpu() < 1] = 0

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
    scale = torch.ones((xyz.shape[0], 3)).cuda().float() * 0.125

    camera_semantic = []
    camera_depth = []
    if not os.path.exists(os.path.join(base_path, sample_token)):
        os.makedirs(os.path.join(base_path, sample_token))
    sem_data_all_path = os.path.join(base_path, sample_token, "semantic.npz")
    depth_data_all_path = os.path.join(base_path, sample_token, "depth_data.npz")

    for cam in cams:
        cam_info = item_data["cams"][cam]
        camera_intrinsic = np.eye(3).astype(np.float32)
        camera_intrinsic[:3, :3] = cam_info["camera_intrinsics"]
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
            sem_save_path = os.path.join(base_path, sample_token, "semantic_color", cam + ".jpg")
            with open(sem_save_path, "wb") as f:
                data = apply_semantic_colormap(render_semantic).cpu().permute(1, 2, 0).detach().numpy() * 255
                f.write(cv2.imencode(".jpg", data)[1])

            if not os.path.exists(os.path.join(base_path, sample_token, "depth_color")):
                os.makedirs(os.path.join(base_path, sample_token, "depth_color"))
            depth_save_path = os.path.join(base_path, sample_token, "depth_color", cam + ".jpg")
            with open(depth_save_path, "wb") as f:
                render_depth = torch.clamp(render_depth, min=0.1, max=40.0)
                data = apply_depth_colormap(render_depth).cpu().permute(1, 2, 0).detach().numpy() * 255
                f.write(cv2.imencode(".jpg", data)[1])

        semantic = torch.max(render_semantic, dim=0)[1].squeeze().cpu().numpy().astype(np.int8)
        camera_semantic.append(semantic)

        depth_data = render_depth[0].detach().cpu().numpy()
        camera_depth.append(depth_data)

    ################################### update object to local ####################################
    np.savez(sem_data_all_path, camera_semantic)
    np.savez(depth_data_all_path, camera_depth)

    print(f"Rendered {sample_token} to {base_path}/{sample_token}")


def load_occ_gt(occ_path, grid_size=np.array([200, 200, 16]), unoccupied=0):
    #  [z y x cls] or [z y x vx vy vz cls]
    pcd = np.load(open(occ_path, "rb"), encoding="bytes", allow_pickle=True)
    pcd_label = pcd[..., -1:]
    pcd_label[pcd_label == 0] = 255
    pcd_np_cor = voxel2world(pcd[..., [0, 1, 2]] + 0.5)  # x y z

    # bevdet augmentation
    pcd_np_cor = world2voxel(pcd_np_cor)

    # make sure the point is in the grid
    pcd_np_cor = np.clip(pcd_np_cor, np.array([0, 0, 0]), grid_size - 1)
    copy.deepcopy(pcd_np_cor)
    pcd_np = np.concatenate([pcd_np_cor, pcd_label], axis=-1)

    # 255: noise, 1-16 normal classes, 0 unoccupied
    pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
    pcd_np = pcd_np.astype(np.int64)
    processed_label = np.ones(grid_size, dtype=np.uint8) * unoccupied
    processed_label = nb_process_label(processed_label, pcd_np)

    noise_mask = processed_label == 255
    processed_label[noise_mask] = 0
    return processed_label


def voxel2world(voxel, voxel_size=np.array([0.5, 0.5, 0.5]), pc_range=np.array([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0])):
    """
    voxel: [N, 3]
    """
    return voxel * voxel_size[None, :] + pc_range[:3][None, :]


def world2voxel(wolrd, voxel_size=np.array([0.5, 0.5, 0.5]), pc_range=np.array([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0])):
    """
    wolrd: [N, 3]
    """
    return (wolrd - pc_range[:3][None, :]) / voxel_size[None, :]


# u1: uint8, u8: uint16, i8: int64
@nb.jit("u1[:,:,:](u1[:,:,:],i8[:,:])", nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)

    return processed_label


def create_full_center_coords(shape=(200, 200, 16), x_range=(-50.0, 50.0), y_range=(-50.0, 50.0), z_range=(-5, 3)):
    x = torch.linspace(x_range[0], x_range[1], shape[0]).view(-1, 1, 1).expand(shape)
    y = torch.linspace(y_range[0], y_range[1], shape[1]).view(1, -1, 1).expand(shape)
    z = torch.linspace(z_range[0], z_range[1], shape[2]).view(1, 1, -1).expand(shape)

    center_coords = torch.stack((x, y, z), dim=-1)

    return center_coords


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--index_list", nargs="+", type=int, default=[0])
    parser.add_argument("--dataset_path", type=str, default="./data/nuScenes/")
    parser.add_argument("--occ_path", type=str, default="./data/occ_data/")
    parser.add_argument("--layout_path", type=str, default="./data/layout/")
    parser.add_argument("--render_path", type=str, default="./data/occ_render_map/")
    parser.add_argument("--vis", action="store_true")

    args = parser.parse_args()

    # render train data
    train_pkl_path = os.path.join(args.dataset_path, "nuscenes_advanced_12Hz_infos_train.pkl")
    render_base_path = os.path.join(args.render_path, "train")
    occ_base_path = os.path.join(args.occ_path, "train")
    layout_base_path = os.path.join(args.layout_path, "train")

    data = pickle.load(open(train_pkl_path, "rb"))
    items_data = data["infos"]

    all_train_items = len(items_data)
    index_list = args.index_list

    for index in index_list:
        try:
            item = items_data[index]
            render_occ_semantic_map(
                item, render_base_path, occ_base_path=occ_base_path, layout_base_path=layout_base_path, is_vis=args.vis
            )

        except Exception as e:
            print(f"Error: {e}")
            with open("./error_list.txt", "a") as f:
                f.write(str(index) + "\n")
            continue

        with open("./success_list.txt", "a") as f:
            f.write(str(index) + "\n")
