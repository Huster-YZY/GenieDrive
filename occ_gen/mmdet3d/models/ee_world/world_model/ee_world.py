import random
import matplotlib.pyplot as plt
import cv2
import copy
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn

from mmdet.models import DETECTORS, BACKBONES
from mmdet.models.utils import build_transformer

from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.models import builder
from mmdet3d.models.losses.lovasz_softmax import lovasz_softmax

from einops import rearrange
import time


def compute_relative_rotation(ego_to_global_rotation):
    """
    :param ego_to_global_rotation: [bs, f, 4]
    :return: [bs, f-1, 4]
    """
    current = ego_to_global_rotation[:, 1:, :]
    previous = ego_to_global_rotation[:, :-1, :]

    previous_conjugate = previous.clone()
    previous_conjugate[:, :, 1:] *= -1

    w1, x1, y1, z1 = previous_conjugate[..., 0], previous_conjugate[..., 1], previous_conjugate[..., 2], \
        previous_conjugate[..., 3]
    w2, x2, y2, z2 = current[..., 0], current[..., 1], current[..., 2], current[..., 3]

    relative_rotation = torch.zeros_like(current)
    relative_rotation[..., 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    relative_rotation[..., 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    relative_rotation[..., 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    relative_rotation[..., 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return relative_rotation


@DETECTORS.register_module()
class EE_World(CenterPoint):
    def __init__(self,
                 # model module
                 vqvae=None,
                 transformer=None,
                 pose_encoder=None,
                 # params
                 previous_frame_exist=False,
                 previous_frame=None,
                 train_future_frame=None,
                 test_future_frame=None,
                 test_previous_frame=None,
                 observe_frame_number=3,
                 memory_frame_number=5,
                 sample_rate=1.0,
                 # Loss
                 feature_similarity_loss=None,
                 recon_loss = None,
                 trajs_loss=None,
                 rotation_loss=None,
                 test_mode=False,
                 e2e_train = False,
                 task_mode='generate',
                 dataset_type='occ3d',
                 eval_time=5,
                 eval_num_frames = 3, 
                 eval_metric='forecasting_miou',
                 class_weights = None,
                 **kwargs):
        super(EE_World, self).__init__(**kwargs)
        # -------- Model Module --------
        self.pose_encoder = builder.build_head(pose_encoder)
        self.transformer = build_transformer(transformer)

        if test_mode or e2e_train:
            self.vqvae = builder.build_detector(vqvae)
        else:
            self.vqvae = builder.build_detector(vqvae)
            self.vqvae.requires_grad_(False)

        self.test_mode = test_mode
        self.e2e = e2e_train

        # -------- Video Params --------
        self.observe_relative_rotation = None
        self.observe_delta_translation = None
        self.observe_ego_lcf_feat = None
        self.task_mode = task_mode

        # -------- Params --------
        self.previous_frame_exist = previous_frame_exist
        self.previous_frame = previous_frame if self.previous_frame_exist else 0
        self.train_future_frame = train_future_frame
        self.test_future_frame = test_future_frame
        self.test_previous_frame = test_previous_frame
        self.observe_frame_number = observe_frame_number + 1  # 2s+current frame default
        self.memory_frame_number = memory_frame_number
        self.sample_rate = sample_rate
        self.dataset_type = dataset_type
        self.eval_time = eval_time
        self.eval_metric = eval_metric

        # -------- Loss -----------
        self.feature_similarity_loss = builder.build_loss(feature_similarity_loss)
        if recon_loss is not None:
            self.recon_loss = builder.build_loss(recon_loss)

        self.trajs_loss = builder.build_loss(trajs_loss)
        self.rotation_loss = builder.build_loss(rotation_loss)
        self.ce_loss = nn.CrossEntropyLoss()
        self.frame_loss_weight = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.05]  # as default
        self.occ_size = [200, 200, 16]
        self.foreground_cls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]
        self.free_cls = 17
        self.save_index = 0

        self.eval_num_frames = eval_num_frames
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights)
            self.ce_loss = nn.CrossEntropyLoss(weight = self.class_weights)
        
        self.cnt = 0
        self.start_save = False

    def obtain_scene_from_token(self, token):
        if token.dim() == 4:
            token = token.unsqueeze(1)

        bs, f, c, w, h = token.shape
        decoder_shapes = [torch.tensor((200, 200)), torch.tensor((100, 100))]
        shapes = (bs, f, 200, 200, 16)
        token = token.view(bs * f, c, w, h)
        # scene = self.vqvae.forward_decoder(token, decoder_shapes, shapes)
        scene = self.vqvae.forward_decoder(token)
        scene = rearrange(scene, 'b c x y z -> b x y z c')
        scene = scene.unflatten(0, (bs, f))
        return scene

    def load_transformation_info(self, img_metas, latent):
        # exit(-1)
        device, dtype = latent.device, latent.dtype

        curr_to_future_ego_rt = torch.stack(
            [torch.tensor(img_meta['curr_to_future_ego_rt'], device=device, dtype=dtype) for img_meta in img_metas])
        curr_ego_to_global_rt = torch.stack(
            [torch.tensor(img_meta['curr_ego_to_global'], device=device, dtype=dtype) for img_meta in img_metas])
        ego_to_global_rotation = torch.stack(
            [torch.tensor(img_meta['ego_to_global_rotation'], device=device, dtype=dtype) for img_meta in
             img_metas])  # quarternion [bs, f, 4]
        ego_to_global_translation = torch.stack(
            [torch.tensor(img_meta['ego_to_global_translation'], device=device, dtype=dtype) for img_meta in img_metas])
        # Change translation to delta translation, only utilize x,y
        ego_to_global_delta_translation = ego_to_global_translation[:, 1:] - ego_to_global_translation[:, :-1]
        ego_to_global_delta_translation = ego_to_global_delta_translation[..., :2]
        # Compute relative rotation
        ego_to_global_relative_rotation = compute_relative_rotation(ego_to_global_rotation)

        if self.dataset_type == 'waymo':
            # use fake gt_ego_lcf_feat and gt_ego_fut_cmd
            future_number = curr_to_future_ego_rt.shape[1]
            gt_ego_lcf_feat = torch.zeros((curr_to_future_ego_rt.shape[0], future_number, 3), device=device,
                                          dtype=dtype)
            gt_ego_fut_cmd = torch.zeros((curr_to_future_ego_rt.shape[0], future_number, 3), dtype=dtype, device=device)
            gt_ego_fut_cmd[..., 0] = 1
        else:
            gt_ego_lcf_feat = torch.stack(
                [torch.tensor(img_meta['gt_ego_lcf_feat'], device=device, dtype=dtype) for img_meta in img_metas])
            gt_ego_fut_cmd = torch.stack(
                [torch.tensor(img_meta['gt_ego_fut_cmd'], device=device, dtype=dtype) for img_meta in img_metas])
            
        start_of_sequence = torch.stack(
            [torch.tensor(img_meta['start_of_sequence'], device=device) for img_meta in img_metas])

        trans_infos = dict(
            curr_to_future_ego_rt=curr_to_future_ego_rt,
            curr_ego_to_global_rt=curr_ego_to_global_rt,
            ego_to_global_rotation=ego_to_global_rotation,
            ego_to_global_translation=ego_to_global_translation,
            ego_to_global_delta_translation=ego_to_global_delta_translation,
            ego_to_global_relative_rotation=ego_to_global_relative_rotation,
            gt_ego_lcf_feat=gt_ego_lcf_feat,
            gt_ego_fut_cmd=gt_ego_fut_cmd,
            # Sequence information
            start_of_sequence=start_of_sequence,
        )
        return trans_infos

    def process_observe_info(self, trans_infos, latent, start_update=True):
        bs, f, c, h, w = latent.shape
        start_of_sequence = trans_infos['start_of_sequence']
        device, dtype = latent.device, latent.dtype
        if start_update:
            if self.observe_relative_rotation is None:
                # Zero-init
                self.observe_relative_rotation = \
                    torch.ones(bs, self.observe_frame_number, 4, device=device, dtype=dtype)
                self.observe_delta_translation = \
                    torch.zeros(bs, self.observe_frame_number, 2, device=device, dtype=dtype)
                self.observe_ego_lcf_feat = \
                    torch.zeros(bs, self.observe_frame_number, 3, device=device, dtype=dtype)

            if start_of_sequence.sum() > 0:
                # Zero-init
                self.observe_relative_rotation[start_of_sequence] = \
                    torch.ones(start_of_sequence.sum(), self.observe_frame_number, 4, device=device, dtype=dtype)
                self.observe_delta_translation[start_of_sequence] = \
                    torch.zeros(start_of_sequence.sum(), self.observe_frame_number, 2, device=device, dtype=dtype)
                self.observe_ego_lcf_feat[start_of_sequence] = \
                    torch.zeros(start_of_sequence.sum(), self.observe_frame_number, 3, device=device, dtype=dtype)

        else:
            self.observe_delta_translation = torch.cat(
                [self.observe_delta_translation[:, 1:], trans_infos['ego_to_global_delta_translation'][:, 0:1]], dim=1)
            self.observe_relative_rotation = torch.cat(
                [self.observe_relative_rotation[:, 1:], trans_infos['ego_to_global_relative_rotation'][:, 0:1]], dim=1)
            self.observe_ego_lcf_feat = torch.cat(
                [self.observe_ego_lcf_feat[:, 1:], trans_infos['gt_ego_lcf_feat'][:, 0:1]], dim=1)

    def init_state(self, trans_infos, latent):
        bs, f, c, h, w = latent.shape
        device, dtype = latent.device, latent.dtype
        # As default memory_frame_number = 5, 4 history frames + 1 current frame
        history_token = latent[:, 0:1].repeat(1, self.memory_frame_number, 1, 1, 1).detach().clone()  # bs, f, c, w, h @zhenya?
        # print(history_token.shape)
        # exit(-1)

        history_ego_lcf_feat = torch.zeros(bs, self.memory_frame_number, 3, device=device, dtype=dtype)
        history_relative_rotation = torch.ones(bs, self.memory_frame_number, 4, device=device, dtype=dtype)
        history_delta_translation = torch.zeros(bs, self.memory_frame_number, 2, device=device, dtype=dtype)

        history_relative_rotation[:, -self.observe_frame_number:] = self.observe_relative_rotation
        history_delta_translation[:, -self.observe_frame_number:] = self.observe_delta_translation
        history_ego_lcf_feat[:, -self.observe_frame_number:] = self.observe_ego_lcf_feat

        #load history latent
        history_token[:, -self.observe_frame_number:] = latent[:, :self.observe_frame_number].detach().clone()

        history_info = dict(
            history_token=history_token,
            history_ego_lcf_feat=history_ego_lcf_feat,
            history_relative_rotation=history_relative_rotation,
            history_delta_translation=history_delta_translation,
        )

        # curr_latent = latent[:, 0].clone()
        curr_latent = latent[:, self.observe_frame_number-1].clone()
        
        # print(trans_infos['curr_to_future_ego_rt'].shape) #[1, 6, 4, 4]
        # exit(-1)

        curr_to_future_ego_rt = trans_infos['curr_to_future_ego_rt'][:, 0].clone()
        curr_ego_to_global = trans_infos['curr_ego_to_global_rt'][:, 0].clone()
        curr_rotation = trans_infos['ego_to_global_rotation'][:, 0].clone()
        curr_translation = trans_infos['ego_to_global_translation'][:, 0].clone()
        curr_ego_lcf_feat = trans_infos['gt_ego_lcf_feat'][:, 0].clone()
        curr_ego_mode = trans_infos['gt_ego_fut_cmd'][:, 0].clone()
        curr_relative_rotation = torch.ones(bs, 4, device=device, dtype=dtype)
        curr_delta_translation = torch.zeros(bs, 2, device=device, dtype=dtype)

        curr_info = dict(
            latent=latent[:, self.observe_frame_number-1:],
            curr_latent=curr_latent,
            curr_to_future_ego_rt=curr_to_future_ego_rt,
            curr_ego_to_global=curr_ego_to_global,
            curr_rotation=curr_rotation,
            curr_translation=curr_translation,
            curr_ego_lcf_feat=curr_ego_lcf_feat,
            curr_ego_mode=curr_ego_mode,
            curr_relative_rotation=curr_relative_rotation,
            curr_delta_translation=curr_delta_translation
        )

        return history_info, curr_info

    def update_curr_info(self, curr_info, trans_infos, pred_trans_info, use_gt_rate, frame_idx, train):
        if self.task_mode == 'generate':
            curr_info['curr_to_future_ego_rt'] = trans_infos['curr_to_future_ego_rt'][:, frame_idx + 1]

        if train:
            curr_latent = torch.zeros_like(curr_info['curr_latent'])
            curr_latent[use_gt_rate] = curr_info['latent'][use_gt_rate, frame_idx + 1]
            curr_latent[~use_gt_rate] = pred_trans_info['pred_latent'][~use_gt_rate]
            curr_info['curr_latent'] = curr_latent

            curr_delta_translation = torch.zeros_like(curr_info['curr_delta_translation'])
            curr_delta_translation[use_gt_rate] = trans_infos['ego_to_global_delta_translation'][use_gt_rate, frame_idx]
            curr_delta_translation[~use_gt_rate] = pred_trans_info['pred_delta_translation'][~use_gt_rate]
            curr_info['curr_delta_translation'] = curr_delta_translation

            curr_translation = torch.zeros_like(curr_info['curr_translation'])
            curr_translation[use_gt_rate] = trans_infos['ego_to_global_translation'][use_gt_rate, frame_idx + 1]
            curr_translation[~use_gt_rate] = pred_trans_info['pred_next_translation'][~use_gt_rate]
            curr_info['curr_translation'] = curr_translation

            curr_rotation = torch.zeros_like(curr_info['curr_rotation'])
            curr_rotation[use_gt_rate] = trans_infos['ego_to_global_rotation'][use_gt_rate, frame_idx + 1]
            curr_rotation[~use_gt_rate] = pred_trans_info['pred_next_rotation'][~use_gt_rate]
            curr_info['curr_rotation'] = curr_rotation

            curr_relative_rotation = torch.zeros_like(curr_info['curr_relative_rotation'])
            curr_relative_rotation[use_gt_rate] = trans_infos['ego_to_global_relative_rotation'][use_gt_rate, frame_idx]
            curr_relative_rotation[~use_gt_rate] = pred_trans_info['pred_relative_rotation'][~use_gt_rate]
            curr_info['curr_relative_rotation'] = curr_relative_rotation

            curr_ego_to_global = torch.zeros_like(curr_info['curr_ego_to_global'])
            curr_ego_to_global[use_gt_rate] = trans_infos['curr_ego_to_global_rt'][use_gt_rate, frame_idx + 1]
            curr_ego_to_global[~use_gt_rate] = pred_trans_info['pred_next_ego_to_global'][~use_gt_rate]
            curr_info['curr_ego_to_global'] = curr_ego_to_global

            curr_ego_lcf_feat = torch.zeros_like(curr_info['curr_ego_lcf_feat'])
            curr_ego_lcf_feat[use_gt_rate] = trans_infos['gt_ego_lcf_feat'][use_gt_rate, frame_idx]
            curr_ego_lcf_feat[~use_gt_rate] = pred_trans_info['pred_ego_lcf_feat'][~use_gt_rate]
            curr_info['curr_ego_lcf_feat'] = curr_ego_lcf_feat

            curr_ego_mode = trans_infos['gt_ego_fut_cmd'][:, frame_idx]
            curr_info['curr_ego_mode'] = curr_ego_mode
        else:
            curr_info['curr_latent'] = pred_trans_info['pred_latent']
            curr_info['curr_delta_translation'] = pred_trans_info['pred_delta_translation']
            curr_info['curr_translation'] = pred_trans_info['pred_next_translation']
            curr_info['curr_relative_rotation'] = pred_trans_info['pred_relative_rotation']
            curr_info['curr_rotation'] = pred_trans_info['pred_next_rotation']
            curr_info['curr_ego_to_global'] = pred_trans_info['pred_next_ego_to_global']
            curr_info['curr_ego_lcf_feat'] = pred_trans_info['pred_ego_lcf_feat']
            curr_ego_mode = trans_infos['gt_ego_fut_cmd'][:, frame_idx]
            curr_info['curr_ego_mode'] = curr_ego_mode
        return curr_info

    def update_history_info(self, history_info, curr_info):
        history_info['history_token'] = torch.cat(
            [history_info['history_token'][:, 1:], curr_info['curr_latent'].unsqueeze(1).detach().clone()], dim=1)
        history_info['history_delta_translation'] = torch.cat(
            [history_info['history_delta_translation'][:, 1:],
             curr_info['curr_delta_translation'].unsqueeze(1).detach().clone()], dim=1)
        history_info['history_relative_rotation'] = torch.cat(
            [history_info['history_relative_rotation'][:, 1:],
             curr_info['curr_relative_rotation'].unsqueeze(1).detach().clone()], dim=1)
        history_info['history_ego_lcf_feat'] = torch.cat(
            [history_info['history_ego_lcf_feat'][:, 1:], curr_info['curr_ego_lcf_feat'].unsqueeze(1).detach().clone()],
            dim=1)
        return history_info

    def forward_sample(self, latent, img_metas, predict_future_frame, train=True, **kwargs):
        # latent: [bs, f, c, h, w]
        bs, f, c, h, w = latent.shape

        # -------------- Load GT Transformation --------------
        trans_infos = self.load_transformation_info(img_metas, latent)

        # ------------- History observe information -------------
        self.process_observe_info(trans_infos, latent, start_update=True)

        # -------------- Init hisotry & input tinformation --------------
        history_info, curr_info = self.init_state(trans_infos, latent) # add history condition (3 history frames + 1 current frame)

        # ---------------- Autogressive prediction ----------------
        pred_latents = []
        pred_relative_rotations, pred_delta_translations = [], []

        for frame_idx in range(predict_future_frame):
            # Decide whether to use GT
            use_gt_rate = torch.rand(size=(bs,), device=latent.device) < self.sample_rate

            plan_query = self.pose_encoder.forward_encoder(history_info)

            pred_trans_info = self.transformer(
                curr_info=curr_info,
                history_info=history_info,
                plan_queries=plan_query,
            )

            pred_trans_info = self.pose_encoder.get_ego_feat(
                pred_trans_info=pred_trans_info,
                curr_info=curr_info,
                start_of_sequence=trans_infos['start_of_sequence']
            )

            if frame_idx != predict_future_frame - 1:
                # Update current info
                curr_info = self.update_curr_info(curr_info, trans_infos, pred_trans_info, use_gt_rate, frame_idx,
                                                  train)
                # update history info
                history_info = self.update_history_info(history_info, curr_info)

            #todo: add res connect

            # Store the intermediate results
            pred_latents.append(pred_trans_info['pred_latent'])
            pred_delta_translations.append(pred_trans_info['pred_delta_translation'])
            pred_relative_rotations.append(pred_trans_info['pred_relative_rotation'])

        # Update observe information
        self.process_observe_info(trans_infos, latent, start_update=False)

        return_dict = dict(
            pred_latents=torch.stack(pred_latents, dim=1),  # [bs, f, c, w, h], pred future latents
            pred_delta_translations=torch.stack(pred_delta_translations, dim=1),  # [bs, f, 2]
            pred_relative_rotations=torch.stack(pred_relative_rotations, dim=1),  # [bs, f, 4], pred future rotations
            targ_delta_translations=trans_infos['ego_to_global_delta_translation'],
            # [bs, f, 2], GT futuredelta translations
            targ_relative_rotations=trans_infos['ego_to_global_relative_rotation'],  # [bs, f, 4], GT future rotations
        )
        return return_dict

    def forward_test(self, **kwargs):
        if self.e2e:
            return self.e2e_forward_test(**kwargs)
        else:
            return self.naive_forward_test(**kwargs)

    def naive_forward_test(self, latent, voxel_semantics, img_metas, **kwargs):
        # Autoregressive predict future latent & Forward future latent
        start_time = time.time()
        # print(latent.shape, voxel_semantics.shape, self.test_future_frame)
        # exit(-1)
        sample_dict = self.forward_sample(latent, img_metas, self.test_future_frame, train=False)
        end_time = time.time()

        return_dict = dict()
        sample_idx = img_metas[0]['sample_idx']
        # Occupancy prediction
        if self.task_mode == 'generate':
            # Forward current latent
            targ_future_voxel_semantics = voxel_semantics[:, self.test_previous_frame + 1:]
            targ_curr_voxel_semantics = voxel_semantics[:, self.test_previous_frame:self.test_previous_frame + 1]
            pred_curr_voxel_semantics = self.obtain_scene_from_token(latent[:, self.test_previous_frame])
            pred_curr_voxel_semantics = pred_curr_voxel_semantics.softmax(-1).argmax(-1)
            # np.save('./vis/naive_pred_0.npy', pred_curr_voxel_semantics.cpu().detach().numpy())

            if self.dataset_type != 'waymo':
                return_dict['pred_curr_semantics'] = pred_curr_voxel_semantics.cpu().numpy().astype(np.uint8)
                return_dict['targ_curr_semantics'] = targ_curr_voxel_semantics.cpu().numpy().astype(np.uint8)

            pred_latents = sample_dict['pred_latents']
            pred_voxel_semantics = self.obtain_scene_from_token(pred_latents)
            pred_voxel_semantics = pred_voxel_semantics.softmax(-1).argmax(-1)
            bs = pred_voxel_semantics.shape[0]

            # for i in range(pred_voxel_semantics.shape[1]):
            #     np.save(f'./vis/naive_pred_{i+1}.npy', pred_voxel_semantics[0][i].cpu().detach().numpy())
            #     # np.save(f'./vis/gt_{i+1}.npy', targ_future_voxel_semantics[0][i].cpu().detach().numpy())
            
            # print(f"Testing {sample_idx}") #Testing 040a956bf0c04ad09fffc286cab5bc56
            # np.save(f'./vis/latent_pred.npy', pred_latents.cpu().detach().numpy())
            # np.save(f'./vis/latent_gt.npy', self.vqvae.encode_to_latent(voxel_semantics.flatten(0,1)).unflatten(0, (bs, -1)).cpu().detach().numpy())
            # exit(-1)

            if self.dataset_type == 'waymo':
                # Due to the large waymo dataset, we only evaluate the eval_time-th frame
                if self.eval_metric == 'forecasting_miou':

                    eval_frame_idxs = [i*2 + 1 for i in range(self.eval_num_frames)]
                    # pred_voxel_semantics = pred_voxel_semantics[:, [1, 3, 5]]
                    # targ_future_voxel_semantics = targ_future_voxel_semantics[:, [1, 3, 5]]

                    pred_voxel_semantics = pred_voxel_semantics[:, eval_frame_idxs]
                    targ_future_voxel_semantics = targ_future_voxel_semantics[:, eval_frame_idxs]

                elif self.eval_metric == 'miou':
                    pred_voxel_semantics = pred_voxel_semantics[:, [self.eval_time]]
                    targ_future_voxel_semantics = targ_future_voxel_semantics[:, [self.eval_time]]

            if self.eval_metric == 'forecasting_miou':
                return_dict['pred_futu_semantics'] = pred_voxel_semantics.cpu().numpy().astype(np.uint8)
                return_dict['targ_futu_semantics'] = targ_future_voxel_semantics.cpu().numpy().astype(np.uint8)
            elif self.eval_metric == 'miou':
                return_dict['semantics'] = pred_voxel_semantics.cpu().numpy().astype(np.uint8)
                return_dict['targ_semantics'] = targ_future_voxel_semantics.cpu().numpy

        # Other information
        return_dict['occ_path'] = [img_meta['occ_path'] for img_meta in img_metas]
        return_dict['occ_index'] = [img_meta['occ_index'] for img_meta in img_metas]
        return_dict['index'] = [img_meta['index'] for img_meta in img_metas]
        return_dict['sample_idx'] = sample_idx
        return_dict['time'] = (end_time - start_time) / self.test_future_frame / bs

        return [return_dict]
    
    def e2e_forward_test(self, voxel_semantics, img_metas, **kwargs):
        # Autoregressive predict future latent & Forward future latent

        # if self.cnt !=200:
        #     self.cnt +=1
        #     return [{}]

        # torch.cuda.synchronize()
        start_time = time.time()
        
        sample_dict = self.e2e_forward_sample(voxel_semantics, img_metas, self.test_future_frame, self.test_previous_frame, train=False)

        # torch.cuda.synchronize()
        end_time = time.time()

        # elapsed_time = end_time - start_time
        # print(f"Time consumed: {elapsed_time:.8f} s")

        return_dict = dict()
        sample_idx = img_metas[0]['sample_idx']

        # Occupancy prediction
        if self.task_mode == 'generate':
            # Forward current latent
            targ_future_voxel_semantics = voxel_semantics[:, self.test_previous_frame + 1:]
            targ_curr_voxel_semantics = voxel_semantics[:, self.test_previous_frame:self.test_previous_frame + 1]

            pred_curr_voxel_semantics = self.obtain_scene_from_token(sample_dict['curr_latent'])
            pred_curr_voxel_semantics = pred_curr_voxel_semantics.softmax(-1).argmax(-1)

            # np.save('./vis/pred_0.npy', pred_curr_voxel_semantics.cpu().detach().numpy())
            # np.save('./vis/gt_0.npy', targ_curr_voxel_semantics.cpu().detach().numpy())

            if self.dataset_type != 'waymo':
                return_dict['pred_curr_semantics'] = pred_curr_voxel_semantics.cpu().numpy().astype(np.uint8)
                return_dict['targ_curr_semantics'] = targ_curr_voxel_semantics.cpu().numpy().astype(np.uint8)

            pred_latents = sample_dict['pred_latents']
            pred_voxel_semantics = self.obtain_scene_from_token(pred_latents)
            pred_voxel_semantics = pred_voxel_semantics.softmax(-1).argmax(-1)
            bs = pred_voxel_semantics.shape[0]

            # if self.cnt ==200:
            pred_list = []
            gt_list = []
            for i in range(pred_voxel_semantics.shape[1]):
                pred_list.append(pred_voxel_semantics[0][i].cpu().detach().numpy())
                gt_list.append(targ_future_voxel_semantics[0][i].cpu().detach().numpy())

            # np.save(f'./vis/iso_{sample_idx}.npy', np.stack(pred_list))

            # np.save(f'./vis/pred_{sample_idx}.npy', np.stack(pred_list))
            # np.save(f'./vis/gt_{sample_idx}.npy', np.stack(gt_list))
            # exit(-1)
            
            # self.cnt +=1

            # print(f"Testing {sample_idx}")
            # np.save(f'./vis/e2e_latent_pred.npy', pred_latents.cpu().detach().numpy())
            # np.save(f'./vis/e2e_latent_gt.npy', self.vqvae.encode_to_latent(voxel_semantics.flatten(0,1)).unflatten(0, (bs, -1)).cpu().detach().numpy())
            # exit(-1)

            if self.dataset_type == 'waymo':
                # Due to the large waymo dataset, we only evaluate the eval_time-th frame
                if self.eval_metric == 'forecasting_miou':
                    pred_voxel_semantics = pred_voxel_semantics[:, [1, 3, 5]]
                    targ_future_voxel_semantics = targ_future_voxel_semantics[:, [1, 3, 5]]
                elif self.eval_metric == 'miou':
                    pred_voxel_semantics = pred_voxel_semantics[:, [self.eval_time]]
                    targ_future_voxel_semantics = targ_future_voxel_semantics[:, [self.eval_time]]

            if self.eval_metric == 'forecasting_miou':
                return_dict['pred_futu_semantics'] = pred_voxel_semantics.cpu().numpy().astype(np.uint8)
                return_dict['targ_futu_semantics'] = targ_future_voxel_semantics.cpu().numpy().astype(np.uint8)
            elif self.eval_metric == 'miou':
                return_dict['semantics'] = pred_voxel_semantics.cpu().numpy().astype(np.uint8)
                return_dict['targ_semantics'] = targ_future_voxel_semantics.cpu().numpy

        # Other information
        return_dict['occ_path'] = [img_meta['occ_path'] for img_meta in img_metas]
        return_dict['occ_index'] = [img_meta['occ_index'] for img_meta in img_metas]
        return_dict['index'] = [img_meta['index'] for img_meta in img_metas]
        return_dict['sample_idx'] = sample_idx
        return_dict['time'] = (end_time - start_time) / self.test_future_frame / bs

        return [return_dict]

    def forward_train(self, **kwargs):
        if self.e2e:
            return self.e2e_forward_train(**kwargs)
        else:
            return self.naive_forward_train(**kwargs)
        
    def naive_forward_train(self, latent, voxel_semantics, img_metas, **kwargs):
        # Forward auto-regressive prediction
        # print(latent.shape, voxel_semantics.shape)
        # exit(-1)

        bs = latent.shape[0]

        return_dict = self.forward_sample(latent, img_metas, self.train_future_frame, train=True)
        pred_latents = return_dict['pred_latents']

        targ_latents = latent[:, self.previous_frame+1:]  # GT future latent

        targ_voxels = voxel_semantics[:, self.previous_frame+1:]
        pred_voxels = self.vqvae.forward_decoder(pred_latents.flatten(0,1)).unflatten(0, (bs, -1))

        # print(pred_voxels.shape, pred_voxels.max(), pred_voxels.min())
        # exit(-1)

        pred_delta_translations = return_dict['pred_delta_translations']
        targ_delta_translations = return_dict['targ_delta_translations']

        pred_relative_rotations = return_dict['pred_relative_rotations']
        targ_relative_rotations = return_dict['targ_relative_rotations']

        # Get valid index for training
        valid_frame = torch.stack(
            [torch.tensor(img_meta['valid_frame'], device=latent.device) for img_meta in img_metas])

        loss_dict = dict()

        for frame_idx in range(self.train_future_frame): #original loss for forecasting training
            loss_dict['feat_sim_{}s_loss'.format((frame_idx + 1) * 0.5)] = self.frame_loss_weight[frame_idx] * \
                                                                           self.feature_similarity_loss(
                                                                               pred_latents[:, frame_idx],
                                                                               targ_latents[:, frame_idx],
                                                                               valid_frame[:, frame_idx])
        
        # for frame_idx in range(self.train_future_frame): #todo: 10 is for e2e tuning
        #     loss_dict['feat_sim_{}s_loss'.format((frame_idx + 1) * 0.5)] = 10.0 * \
        #                                                                    self.feature_similarity_loss(
        #                                                                        pred_latents[:, frame_idx],
        #                                                                        targ_latents[:, frame_idx],
        #                                                                        valid_frame[:, frame_idx])
        
        loss_dict['trajs_loss'] = self.trajs_loss(pred_delta_translations, targ_delta_translations, valid_frame, None)
        loss_dict['rotation_loss'] = self.rotation_loss(pred_relative_rotations, targ_relative_rotations, valid_frame,
                                                        None)
            
        # for frame_idx in range(self.train_future_frame): #todo: do we need weighting like feat_similarity_loss?
        #     loss_dict['recon_{}s_loss'.format((frame_idx + 1) * 0.5)] = self.e2e_reconstruct_loss(pred_voxels[:, frame_idx], targ_voxels[:, frame_idx], valid_frame[:, frame_idx])

        return loss_dict
    
    def e2e_forward_train(self, voxel_semantics, img_metas, **kwargs):
        # End-to-End Forward auto-regressive prediction

        # print(voxel_semantics.shape) #[8, 7, 200, 200, 16]
        # print(img_metas.shape)
        # exit(-1)
        
        return_dict = self.e2e_forward_sample(voxel_semantics, img_metas, self.train_future_frame, self.previous_frame, train=True)

        pred_latents = return_dict['pred_latents']
        targ_latents = return_dict['targ_latents']

        assert pred_latents.shape == targ_latents.shape

        bs = pred_latents.shape[0]
        
        targ_voxels = voxel_semantics[:, self.previous_frame+1:]
        pred_voxels = self.vqvae.forward_decoder(pred_latents.flatten(0,1)).unflatten(0, (bs, -1))


        pred_delta_translations = return_dict['pred_delta_translations']
        targ_delta_translations = return_dict['targ_delta_translations']

        pred_relative_rotations = return_dict['pred_relative_rotations']
        targ_relative_rotations = return_dict['targ_relative_rotations']

        # Get valid index for training
        valid_frame = torch.stack([torch.tensor(img_meta['valid_frame'], device=voxel_semantics.device) for img_meta in img_metas])

        loss_dict = dict()

        # for frame_idx in range(self.train_future_frame): #todo: 10 is for e2e tuning
        #     loss_dict['feat_sim_{}s_loss'.format((frame_idx + 1) * 0.5)] = 10.0 * \
        #                                                                    self.feature_similarity_loss(
        #                                                                        pred_latents[:, frame_idx],
        #                                                                        targ_latents[:, frame_idx],
        #                                                                        valid_frame[:, frame_idx])
        
        loss_dict['trajs_loss'] = self.trajs_loss(pred_delta_translations, targ_delta_translations, valid_frame, None)
        loss_dict['rotation_loss'] = self.rotation_loss(pred_relative_rotations, targ_relative_rotations, valid_frame, None)
            
        for frame_idx in range(self.train_future_frame): #todo: do we need weighting like feat_similarity_loss?
            loss_dict['recon_{}s_loss'.format((frame_idx + 1) * 0.5)] = self.e2e_reconstruct_loss(pred_voxels[:, frame_idx], targ_voxels[:, frame_idx], valid_frame[:, frame_idx])

        return loss_dict

    def e2e_reconstruct_loss(self, pred, targ, valid):
        assert hasattr(self, 'recon_loss')

        valid_pred = pred[valid]
        valid_targ = targ[valid]

        if valid_pred.shape[0] == 0:
            return pred.sum()*0.0 #torch.tensor(0.0, device=pred.device, dtype=pred.dtype) # may lead to "receive no gradient"
        

        # loss_reconstruct = self.recon_loss(pred=valid_pred, target=valid_targ)
        loss_reconstruct = self.ce_loss(valid_pred, valid_targ)

        valid_pred = torch.softmax(valid_pred, dim=1)
        loss_lovasz = lovasz_softmax(valid_pred, valid_targ, ignore=255)
        # print(loss_reconstruct, loss_lovasz)

        return loss_reconstruct + loss_lovasz


    def e2e_forward_sample(self, voxel, img_metas, predict_future_frame, previous_frame, train=True, **kwargs):
        # latent: [bs, f, c, h, w]
        # bs, f, c, h, w = latent.shape
        bs = voxel.shape[0]
        frame_num = voxel.shape[1]
        
        latent = self.vqvae.encode_to_latent(voxel.flatten(0,1)).unflatten(0, (bs, -1))

        # -------------- Load GT Transformation --------------
        trans_infos = self.load_transformation_info(img_metas, latent)

        # ------------- History observe information -------------
        self.process_observe_info(trans_infos, latent, start_update=True)

        # -------------- Init hisotry & input tinformation --------------
        history_info, curr_info = self.init_state(trans_infos, latent)

        # ---------------- Autogressive prediction ----------------
        pred_latents = []
        pred_relative_rotations, pred_delta_translations = [], []

        for frame_idx in range(predict_future_frame):
            # Decide whether to use GT
            use_gt_rate = torch.rand(size=(bs,), device=latent.device) < self.sample_rate

            plan_query = self.pose_encoder.forward_encoder(history_info)

            pred_trans_info = self.transformer(
                curr_info=curr_info,
                history_info=history_info,
                plan_queries=plan_query,
            )

            pred_trans_info = self.pose_encoder.get_ego_feat(
                pred_trans_info=pred_trans_info,
                curr_info=curr_info,
                start_of_sequence=trans_infos['start_of_sequence']
            )

            if frame_idx != predict_future_frame - 1:
                # Update current info
                curr_info = self.update_curr_info(curr_info, trans_infos, pred_trans_info, use_gt_rate, frame_idx,
                                                  train)
                # update history info
                history_info = self.update_history_info(history_info, curr_info)

            # Store the intermediate results
            pred_latents.append(pred_trans_info['pred_latent'])
            pred_delta_translations.append(pred_trans_info['pred_delta_translation'])
            pred_relative_rotations.append(pred_trans_info['pred_relative_rotation'])

        # Update observe information
        self.process_observe_info(trans_infos, latent, start_update=False)

        return_dict = dict(
            pred_latents=torch.stack(pred_latents, dim=1),  # [bs, f, c, w, h], pred future latents
            targ_latents = latent[:, previous_frame+1:],
            curr_latent = latent[:, previous_frame:previous_frame+1],
            pred_delta_translations=torch.stack(pred_delta_translations, dim=1),  # [bs, f, 2]
            pred_relative_rotations=torch.stack(pred_relative_rotations, dim=1),  # [bs, f, 4], pred future rotations
            targ_delta_translations=trans_infos['ego_to_global_delta_translation'],
            # [bs, f, 2], GT futuredelta translations
            targ_relative_rotations=trans_infos['ego_to_global_relative_rotation'],  # [bs, f, 4], GT future rotations
        )
        return return_dict