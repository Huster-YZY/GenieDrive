import os.path
import math

import cv2
import copy
import time

import mmcv
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat

from mmdet.models import DETECTORS

from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.models import builder

from mmdet3d.models.losses.lovasz_softmax import lovasz_softmax


def frequency_embedding(x, dim, max_period: int = 10000):
    bs = x.shape[0]
    x = x.flatten(0, 1)

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=x.device)
    
    args = x[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    embedding = rearrange(embedding, '(b c) d -> b (c d)', b=bs)
    return embedding

class CNN_encoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CNN_encoder, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv3d(in_channels=dim_in, out_channels=dim_out, kernel_size=5, stride=2, padding=2)  
        self.conv2 = nn.Conv3d(in_channels=dim_out, out_channels=dim_out, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x
    
class CNN_decoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CNN_decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(in_channels=dim_in, out_channels=dim_out, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(in_channels=dim_out, out_channels=dim_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)

    def forward(self, v, k, q):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        # Scaled Dot-Product Attention
        attention_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.depth ** 0.5)

        attention_weights = F.softmax(attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)

        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        return self.dense(output)
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff = None, dropout_rate = 0.5):
        super(TransformerBlock, self).__init__()
        if d_ff is None:
            d_ff = d_model

        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)

        self.ffn = FeedForward(d_model, d_ff, dropout_rate)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        return self.layernorm2(x + self.dropout2(ffn_output))
    
@DETECTORS.register_module()
class TriplaneVAE(CenterPoint):
    def __init__(self,
                 occ_shape, 
                 num_classes, 
                 expansion=8, 
                 d_model=16, 
                 num_heads=2, 
                 depth=2, 
                 d_ff=None, 
                 dropout_rate=0.0,
                 **kwargs):
        super(TriplaneVAE, self).__init__(**kwargs)
        if d_ff is None:
            d_ff = d_model
        self.num_classes = num_classes
        
        self.class_embeds = nn.Embedding(num_classes, expansion) #todo: language code book initialization

        X, Y, Z = occ_shape
        downsample_rate = 4 #todo: more elegant implementation sync with encoder & decoder
        shape_dict = dict(x=X//downsample_rate, y=Y//downsample_rate, z=Z//downsample_rate)
        plane_names = ['xy', 'xz', 'yz']
        reduce_dim = {'xy': 'z', 'xz': 'y', 'yz': 'x'}
        self.reduce_dim = reduce_dim
        self.shape_dict = shape_dict

        self.tokens = nn.ParameterDict()
        self.pos_encodings = nn.ParameterDict()
        self.transformers = nn.ModuleDict()
        self.rearrange1 = dict()
        self.rearrange2 = dict()

        for plane in plane_names:
            self.tokens[plane] = nn.Parameter(torch.randn(1, 1, d_model))
            self.pos_encodings[plane] = nn.Parameter(torch.randn(1, shape_dict[reduce_dim[plane]] + 1, d_model))
            # self.transformers[plane] = nn.Sequential(*[TransformerBlock(d_model*2, num_heads, d_ff, dropout_rate) for _ in range(depth)])
            self.transformers[plane] = nn.Sequential(*[TransformerBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(depth)])
            self.rearrange1[plane] = (f'b c x y z -> 'f'(b {" ".join(plane)}) {reduce_dim[plane]} c')  # bcxyz->(bxy)zc
            self.rearrange2[plane] = (f'(b {" ".join(plane)}) c -> b c {" ".join(plane)}', {key: shape_dict[key] for key in plane})
            

        self.conv = nn.Conv3d(expansion, d_model, kernel_size=1)
        # self.vox_encoder = CNN_encoder(dim=d_model) #channel mult*2
        # self.vox_decoder = CNN_decoder(dim=d_model)
        self.vox_encoder = CNN_encoder(dim_in=d_model, dim_out = d_model)

        
        self.vox_decoder = CNN_decoder(dim_in=d_model, dim_out = d_model)

        #vae
        self.fc_mu_logvar = nn.ModuleList([nn.Linear(d_model, d_model * 2) for _ in range(3)]) #todo: reduce latent dim

        #occ decoder
        self.decode_pos_embed = nn.Parameter(torch.randn(1, 1, X//downsample_rate, Y//downsample_rate, Z//downsample_rate))
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, self.num_classes),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.num_classes, self.num_classes)
        )

    def forward_encoder(self, voxel_semantics):
        x = voxel_semantics
        x = self.class_embeds(x)
        x = rearrange(x, 'b x y z c-> b c x y z').contiguous()
        x = self.conv(x)
        x = self.vox_encoder(x)

        xy = self.forward_plane(x, 'xy')
        xz = self.forward_plane(x, 'xz')
        yz = self.forward_plane(x, 'yz')

        #add vae sample
        mus, logvars = list(), list()
        for i, plane in enumerate([xy, xz, yz]):
            _, _, dim1, dim2 = plane.shape
            plane = rearrange(plane, 'b c d1 d2 -> b (d1 d2) c')
            mu_logvar = self.fc_mu_logvar[i](plane)
            mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
            mus.append(rearrange(mu, 'b (d1 d2) c -> b c d1 d2', d1=dim1, d2=dim2))
            logvars.append(rearrange(logvar, 'b (d1 d2) c -> b c d1 d2', d1=dim1, d2=dim2))

        xy, xz, yz = [self.latent_sample(mus[i], logvars[i]) for i in range(3)]
        return torch.cat([xy, xz, yz], dim=-1)

    def forward_decoder(self, latent):
        xy, xz, yz = torch.split(latent, [self.shape_dict['y'], self.shape_dict['z'], self.shape_dict['z']], dim=-1)
        xy = repeat(xy, 'b c x y -> b c x y z', z=self.shape_dict['z'])
        xz = repeat(xz, 'b c x z -> b c x y z', y=self.shape_dict['y'])
        yz = repeat(yz, 'b c y z -> b c x y z', x=self.shape_dict['x'])
        vox_ft = xy * xz * yz  + self.decode_pos_embed
        vox_ft = self.vox_decoder(vox_ft)
        vox_ft = rearrange(vox_ft, 'b c x y z -> b x y z c')
        x = self.cls_head(vox_ft)
        x = rearrange(x, 'b x y z c -> b c x y z')
        return x
    
    def forward_plane(self, x, plane_name):
        x = rearrange(x, self.rearrange1[plane_name])
        token = repeat(self.tokens[plane_name], '1 1 c -> b 1 c', b=x.shape[0])
        plane = torch.cat([x, token], dim=1)
        plane = plane + self.pos_encodings[plane_name]
        plane = self.transformers[plane_name](plane)[:, 0]  # B*X*Y, 1, C
        plane = rearrange(plane, self.rearrange2[plane_name][0], **self.rearrange2[plane_name][1]) # B, X, Y, C
        return plane
    
    def latent_sample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu



    # def forward_decoder(self, z):
    #     return self.decoder(z)

    def forward_test(self, voxel_semantics, img_metas, **kwargs):
        exit(-1)
        # 0. Prepare Input
        bs, t, w, h, d = voxel_semantics.shape

        # 2. Process current voxel semantics
        start_time = time.time()
        curr_voxel, shapes = self.forward_encoder(voxel_semantics.flatten(0,1))
        # print(curr_voxel)
        # exit(-1)

        # 3. Time fusion
        curr_triplane = self.encoder.volume_2_triplane(curr_voxel)
        curr_bev = curr_triplane[..., :50]
        sampled_bev = self.align_bev(curr_bev, img_metas)
        sampled_triplane = torch.zeros((sampled_bev.shape[0], sampled_bev.shape[1], sampled_bev.shape[2], sampled_bev.shape[3], 58), dtype=sampled_bev.dtype, device=sampled_bev.device)
        sampled_triplane[...,:50] = sampled_bev

        # 4. vq
        z_sampled, loss, info = self.vq(curr_triplane, sampled_triplane, is_voxel=False)
        end_time = time.time()

        # 5. Process Decoder
        logits = self.forward_decoder(z_sampled)

        # 6. Preprocess logits
        output_dict = dict()
        pred = logits.softmax(1).argmax(1).cpu().numpy()

        sub_dir = 'token_reg'
        if self.save_results:
            # z_sampled: [bs, c, h, w]
            save_token = z_sampled[0].cpu().numpy()
            if self.results_type != 'waymo':
                mmcv.mkdir_or_exist(os.path.join(self.save_root, sub_dir, str(img_metas[0]['scene_name'])))
                np.savez(os.path.join(self.save_root, sub_dir, str(img_metas[0]['scene_name']), '{}.npz'.format(img_metas[0]['sample_idx'])), token=save_token)
            else:
                occ_path_idx = img_metas[0]['occ_path'].split('/')[-1].split('.')[0]
                mmcv.mkdir_or_exist(os.path.join(self.save_root, sub_dir, str(img_metas[0]['scene_name']).zfill(3)))
                np.savez(os.path.join(self.save_root, sub_dir, str(img_metas[0]['scene_name']).zfill(3), '{}.npz'.format(occ_path_idx)), token=save_token)

            # save pred
            # mmcv.mkdir_or_exist('save_dir/debug/{}'.format(img_metas[0]['scene_name']))
            # np.savez('save_dir/debug/{}/{}.npz'.format(img_metas[0]['scene_name'],img_metas[0]['sample_idx']), semantics=pred[0][0])

        output_dict['semantics'] = pred.astype(np.uint8)
        output_dict['targ_semantics'] = voxel_semantics.cpu().numpy().astype(np.uint8)
        output_dict['index'] = [img_meta['index'] for img_meta in img_metas]
        output_dict['time'] = end_time - start_time
        return [output_dict]

    def forward_train(self, voxel_semantics, img_metas, **kwargs):
        exit(-1)
        # 0. Prepare Input
        bs, t, w, h, d = voxel_semantics.shape  # t=1

        # 2. Process current voxel semantics
        t0 = time.time()
        curr_voxel, shapes = self.forward_encoder(voxel_semantics.flatten(0,1))
        # print(curr_voxel)
        # exit(-1)

        # 3. Time fusion -> Interpolated 3D Occ
        t1 = time.time()
        

        curr_triplane = self.encoder.volume_2_triplane(curr_voxel)
        curr_bev = curr_triplane[..., :50]
        sampled_bev = self.align_bev(curr_bev, img_metas)
        sampled_triplane = torch.zeros((sampled_bev.shape[0], sampled_bev.shape[1], sampled_bev.shape[2], sampled_bev.shape[3], 58), dtype=sampled_bev.dtype, device=sampled_bev.device)
        sampled_triplane[...,:50] = sampled_bev


        # sampled_triplane = self.encoder.volume_2_triplane(sampled_voxel.flatten(0,1)).unflatten(0, (bs, -1))
        t2 = time.time()

        # voxel = torch.cat([curr_voxel, sampled_voxel.flatten(0,1)], dim=0)
        # triplane = self.encoder.volume_2_triplane(voxel)
        # curr_triplane = triplane[:bs]
        # sampled_triplane = triplane[bs:].unflatten(0, (bs, -1))

        # 4. vq
        t3 = time.time()
        z_sampled, loss, info = self.vq(curr_triplane, sampled_triplane, is_voxel=False)

        
        if not self.xo_training:
            p2c_r = []
            p2c_t = []

            for img_meta in img_metas:
                p2c_mat = np.linalg.inv(img_meta['curr_to_prev_ego_rt']) #c2p -> p2c
                p2c_r.append(torch.tensor(p2c_mat[:3, :3].reshape(-1), dtype=torch.float32))
                p2c_t.append(torch.tensor(p2c_mat[:3, 3], dtype=torch.float32))

            p2c_r = torch.stack(p2c_r).to(z_sampled)
            p2c_t = torch.stack(p2c_t).to(z_sampled)
            p2c_rt = torch.cat([p2c_r, p2c_t], dim=-1)

            p2c_embed = frequency_embedding(p2c_rt, dim = 8)
            mean_velocity = self.mean_velocity_net(p2c_embed)


        # eject noise into sampeld token for better generalization
        noise = torch.randn_like(z_sampled) * 0.01
        z_sampled = z_sampled + noise

        # 5. Process Decoder
        t4 = time.time()
        logits = self.forward_decoder(z_sampled)
        t5 = time.time()

        # print(f"Voxel Encoder {t1-t0}\n Align Voxel {t2-t1}\n Triplane {t3-t2}\n VQ {t4-t3}\n Decoder {t5-t4}")

        # 6. Compute Loss
        loss_dict = dict()
        loss_dict.update(self.reconstruct_loss(logits, voxel_semantics))
        loss_dict['embed_loss'] = self.embed_loss_weight * loss

        start_of_sequence = np.array([img_meta['start_of_sequence'] for img_meta in img_metas])
        if self.prev_latent is None:
            self.prev_latent = z_sampled.clone()
        if start_of_sequence.sum() > 0:
            self.prev_latent[start_of_sequence] = z_sampled[start_of_sequence]
        self.prev_latent = self.prev_latent.detach()

        # print((self.prev_latent - z_sampled).shape)
        # print(torch.mean(self.prev_latent - z_sampled, dim=[2,3]).shape)
        # exit(-1)

        if not self.xo_training:
            loss_dict['trans_reg_loss'] = self.trans_reg_loss_weight * F.mse_loss(torch.mean(z_sampled - self.prev_latent, dim=[2,3]), mean_velocity)
        
        self.prev_latent = z_sampled.detach().clone()

        return loss_dict
    
    def encode_to_latent(self, voxel_semantics):
        return self.forward_encoder(voxel_semantics)

