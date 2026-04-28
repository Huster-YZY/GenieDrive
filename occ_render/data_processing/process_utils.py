import os
import cv2
import json
import torch
import pickle
import shutil
import argparse
import numpy as np
from tqdm import tqdm

color_id = np.zeros((20, 3), dtype=np.uint8)
color_id[0, :] = [255, 120, 50]
color_id[1, :] = [255, 192, 203]
color_id[2, :] = [255, 255, 0]
color_id[3, :] = [0, 150, 245]
color_id[4, :] = [0, 255, 255]
color_id[5, :] = [255, 127, 0]
color_id[6, :] = [255, 0, 0]
color_id[7, :] = [255, 240, 150]
color_id[8, :] = [135, 60, 0]
color_id[9, :] = [160, 32, 240]
color_id[10, :] = [255, 0, 255]
color_id[11, :] = [139, 137, 137]
color_id[12, :] = [75, 0, 75]
color_id[13, :] = [150, 240, 80]
color_id[14, :] = [230, 230, 250]
color_id[15, :] = [0, 175, 0]
color_id[16, :] = [0, 255, 127]
color_id[17, :] = [222, 155, 161]
color_id[18, :] = [140, 62, 69]
color_id[19, :] = [227, 164, 30]

def output_video(image_paths, output_video_path):
    fps = 12
    first_image = cv2.imread(image_paths[0])
    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for image_path in image_paths:
        image = cv2.imread(image_path)
        video_writer.write(image)
    video_writer.release()

# def output_sem_video(sem_list, output_path, first_ref = False):
#     new_frames = []
#     for i, array in enumerate(sem_list):
#         if first_ref and i==0:
#             new_frames.append(array)
#             continue
#         for _ in range(6): #repeat 2Hz infos 6 times--> 12Hz conditions for video generation
#             new_frames.append(array)
#     fps = 12
#     height, width, _ = new_frames[0].shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#     for frame in new_frames:
#         # print(type(frame), frame.shape, frame)
#         out.write(frame)
#     out.release()

def output_sem_video(sem_list, output_path, first_ref=False, repeat=6, fps=12):
    """
    Save semantic frames as a video.

    Args:
        sem_list: list of H x W x 3 uint8 frames.
        output_path: output mp4 path.
        first_ref: if True, write the first frame only once.
        repeat: repeat each semantic frame this many times.
        fps: output video FPS.
    """
    if len(sem_list) == 0:
        raise ValueError("sem_list is empty, cannot write video.")

    first_frame = sem_list[0]

    if first_frame.ndim != 3 or first_frame.shape[2] != 3:
        raise ValueError(f"Expected frame shape H x W x 3, but got {first_frame.shape}")

    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for i, frame in enumerate(sem_list):
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame {i} has shape {frame.shape}, expected height={height}, width={width}"
                )

            # OpenCV VideoWriter expects uint8 and contiguous memory.
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            frame = np.ascontiguousarray(frame)

            num_repeat = 1 if first_ref and i == 0 else repeat

            for _ in range(num_repeat):
                out.write(frame)

    finally:
        out.release()

def apply_semantic_colormap(semantic):
    """
    Input:
        semantic: semantic image, tensor/numpy, (N, H, W)
    Output:
        depth: (3, H, W), tensor
    """
    if semantic.shape[0] != 1:
        semantic = torch.max(semantic, dim=0)[1].squeeze()
    else:
        semantic = semantic.squeeze()

    x = torch.zeros((3, semantic.shape[0], semantic.shape[1]), dtype=torch.float)
    for i in range(20):
        x[0][semantic == i] = color_id[i][0]
        x[1][semantic == i] = color_id[i][1]
        x[2][semantic == i] = color_id[i][2]

    return x / 255.0

cam_names = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
cam_dict = {
        "CAM_FRONT_LEFT": 5,
        "CAM_FRONT": 0,
        "CAM_FRONT_RIGHT": 1,
        "CAM_BACK_RIGHT": 2,
        "CAM_BACK": 3,
        "CAM_BACK_LEFT": 4,
    }