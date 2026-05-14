import os
import torch
import random
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from utils.dist_utils import distributed
from pyquaternion import Quaternion

def read_occ(file_path, vox_res, num_classes):
    if ".npy" in file_path:
        occ = np.load(file_path).reshape([128,128,8])
    else:
        occ = np.fromfile(file_path, dtype=np.uint32).reshape(vox_res)
    one_hot = torch.nn.functional.one_hot(torch.tensor(occ.astype(np.int64)), num_classes=num_classes)
    x = one_hot.permute(3, 0, 1, 2).to(torch.float) # C, X, Y, Z
    return x

def parse_semantic_dict(semantic_dict):
    max_key = max(semantic_dict.keys())
    array = np.full(max_key + 1, 0, dtype=np.int64)
    for key, value in semantic_dict.items():
        array[key] = value
    return array

class OccNuscenesDataset(Dataset):
    def __init__(self, data_path, cfg_path, transform = None, train = True, return_path = False):
        super(OccNuscenesDataset, self).__init__()
        
        import json, yaml
        with open(os.path.join(data_path, 'annotations.json'), 'r') as f:
            json_file = json.load(f)
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            self.num_classes = self.cfg["num_classes"]
            self.semantic_names = self.cfg["semantic_names"]

        self.scene_names = json_file["train_split"] if train else json_file["val_split"]
        self.sequence = self.get_sequence(data_path)
        self.data_path = data_path
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        label_file = self.sequence[idx]
        label = np.load(label_file)
        occ = torch.tensor(label['semantics'].astype(np.int64))
        if self.return_path:
            return occ, label_file
        return occ
    
    def get_sequence(self, data_path):
        sequence = []
        base_path = os.path.join(data_path, 'gts')
        for scene in self.scene_names:
            scene_path = os.path.join(base_path, scene)
            vox_paths = [os.path.join(scene_path, f, "labels.npz") for f in os.listdir(scene_path)]
            sequence.extend(vox_paths)
        return sequence
    
    def compute_latent_stats(self):
        num_samples = min(10000, len(self.sequence))
        random_indices = np.random.choice(len(self.sequence), num_samples, replace=False)
        latents = []
        for idx in tqdm(random_indices):
            latent_path = self.sequence[idx]
            pt_path = latent_path.replace("labels.npz", "latent.pt")
            feature = torch.load(pt_path, map_location="cpu")
            latents.append(feature)
        # latents = torch.cat(latents, dim=0)
        latents = torch.stack(latents)
        print(latents.shape)
        mean = latents.mean(dim=[0, 2, 3], keepdim=True)
        std = latents.std(dim=[0, 2, 3], keepdim=True)
        latent_stats = {'mean': mean, 'std': std}
        print(latent_stats)
        return latent_stats
     
def quaternion_to_yaw(q):
    w, x, y, z = q[:,0], q[:,1], q[:, 2], q[:, 3]
    yaw = np.arctan2(2.0 * (w*z+x*y), 1-2*(y**2+z**2))
    return yaw

def yaw2rotmat(yaw):
    cos_yaw = np.cos(yaw).reshape(-1,1)
    sin_yaw = np.sin(yaw).reshape(-1,1)
    R = np.hstack([cos_yaw, -sin_yaw, sin_yaw, cos_yaw]).reshape(-1,2,2)
    return R

def tf_mat(rot, t):
    B = rot.shape[0]
    assert t.shape[0] == B
    res = np.zeros((B, 3, 3), dtype = np.float32)
    res[:, :2, :2] = rot
    res[:, :2, 2] = t
    res[:, 2, 2] = 1
    return res

def get_meta_data(poses):
    from pyquaternion import Quaternion
    rel_pose = np.linalg.inv(poses[:-1]) @ poses[1:]
    rel_pose=  np.concatenate([rel_pose,rel_pose[-1:]], axis=0)
    xyzs = rel_pose[:, :3, 3]

    xys = xyzs[:, :2]
    e2g_t = poses[:, :3, 3]
    # rot 2 quat
    e2g_r = np.array([Quaternion(matrix=pose[:3, :3],atol=1e-7).elements for pose in poses])
    rel_yaws = Rotation.from_matrix(rel_pose[:,:3,:3]).as_euler('zyx', degrees=False)[:,0]

    #get traj (not velocity)  relative to first frame
    e2g_rel0_t = e2g_t.copy()
    # Convert rotations to rotation matrices
    e2g_r_w_last = e2g_r.copy()
    e2g_r_w_last[:, [0, 1, 2 ,3]] = e2g_r_w_last[:, [1, 2,3, 0]] 
    r0 = Rotation.from_quat(e2g_r_w_last[0]).as_matrix()  # First rotation matrix
    rotations = Rotation.from_quat(e2g_r_w_last).as_matrix()  # All rotation matrices
    e2g_rel0_t = np.linalg.inv(r0) @ ( e2g_t - e2g_t[0]).T
    e2g_rel0_t = e2g_rel0_t.T

    rr=np.array([
        [0,-1],
        [1,0],]
    )
    xys=xys@rr.T
    rel_poses_yaws=np.concatenate([xys,rel_yaws[:,None]],axis=1)
    
    return {
        'rel_poses': xys,
        'rel_poses_xyz': xyzs,
        'e2g_t': e2g_t,
        'e2g_r': e2g_r,
        'rel_poses_yaws':rel_poses_yaws,
        'e2g_rel0_t':e2g_rel0_t
    }
    

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(dataset, batch_size=32, num_workers=4, seed=0, drop_last=False):
    generator = torch.Generator()
    generator.manual_seed(seed)

    if distributed():
        sampler = DistributedSampler(dataset, seed=seed)
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=drop_last,
        pin_memory=True,
    )

def get_OccNuscenes_dataloader(data_path, cfg_path, batch_size=32, train=True, return_path=False, load_latent=False):
    transform = None
    dataset = OccNuscenesDataset(data_path, cfg_path, transform=transform, train=train, return_path=return_path)
    dataloader = create_dataloader(dataset, batch_size = batch_size)
    
    return dataloader, dataset.num_classes, (200,200,16)