import torch
import os
import torch.nn as nn
import numpy as np
from vae import OccupancyVAE
from utils.dist_utils import *
from einops import rearrange

from utils.occ_metrics import Metric_mIoU
from dataset import read_occ, get_OccNuscenes_dataloader

set_tf32(True)
rank, device = setup_dist(verbose=True)

def load_vae(model_path, d_model, device):
    model = OccupancyVAE((200,200,16), num_classes = 18, d_model=d_model).to(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model

@torch.no_grad()
def save_vae_latent(device, data_path, cfg_path, is_train = True, vae_ckpt_path = None):
    from tqdm import tqdm
    dataloader, num_classes, occ_shape = get_OccNuscenes_dataloader(data_path, cfg_path, batch_size=32, train=is_train, return_path=True, load_latent=False)
    model = load_vae(model_path = vae_ckpt_path, d_model=64, device=device)
    model.eval()

    pbar = tqdm(dataloader)
    for _, (x, batch_path) in enumerate(pbar):
        x = x.to(device)
        with torch.no_grad():
            outputs = model.encode(x, concat=True).cpu()

        for latent, path in zip(outputs, batch_path):
            save_path = path.replace("gts", "save_dir/token_vae")
            save_path = save_path.replace("/labels.npz", ".npz")
            directory = os.path.dirname(save_path)
    
            if not os.path.exists(directory):
                os.makedirs(directory)
            np.savez(save_path, token = latent.numpy())

if __name__ == "__main__":
    data_path = '../data/nuscenes'
    cfg_path = 'configs/dataset.yaml'
    save_vae_latent(device, data_path, cfg_path, is_train = True, vae_ckpt_path = '../ckpts/vae_d64_210.pth')
    save_vae_latent(device, data_path, cfg_path, is_train = False, vae_ckpt_path = '../ckpts/vae_d64_210.pth')
    cleanup_dist()
