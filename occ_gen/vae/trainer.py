import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict

from utils.dist_utils import distributed, func_rank_0, rank_0
from utils.lovasz import lovasz_softmax
from utils.focal_loss import FocalLoss
from utils.occ_metrics import Metric_mIoU

from dataset import get_OccNuscenes_dataloader
from vae import OccupancyVAE

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# from torch.amp import GradScaler, autocast #for new torch vision
from torch.cuda.amp import GradScaler, autocast #for old torch vision

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
    
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

class VAETrainer:
    def __init__(self, device, data_path, cfg_path, dim_latent = 16, batch_size=16, use_focal_loss = False, dropout_rate=0.5):
        learning_rate = 1e-4
        self.device = device
        # self.model = OccupancyVAE((128, 128, 8)).to(self.device) #CarlaSC
        # self.dataloader = get_CarlaSC_dataloader(batch_size=batch_size) #CarlaSC
        self.data_path, self.cfg_path = data_path, cfg_path
        self.dataloader, num_classes, occ_shape = get_OccNuscenes_dataloader(data_path=data_path, cfg_path=cfg_path, batch_size=batch_size, train=True)
        self.model = OccupancyVAE(occ_shape, num_classes, d_model=dim_latent, num_heads=2, dropout_rate=dropout_rate).to(self.device)
        if distributed():
            self.model = DDP(self.model, device_ids=[self.device])

        self.ce_loss = FocalLoss() if use_focal_loss else nn.CrossEntropyLoss()
        self.kl_loss = self._get_kl_loss()
        self.lovasz_loss = self._get_lovasz_function()
        self.kl_coeff = 0.005
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epoch_base = 0
        self.batch_size = batch_size
        self.scaler = GradScaler()

    def _get_kl_loss(self):
        def _kl_loss(mu, logvar):
            return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return lambda mus, logvars: (sum([_kl_loss(mu, logvar) for mu, logvar in zip(mus, logvars)]) /
                                 sum([np.prod(list(mu.shape)) for mu in mus]))
    
    def _get_lovasz_function(self, **kwargs):
        """
        pred: -1, num_classes
        gt: -1,
        """
        return lambda pred, gt: lovasz_softmax(F.softmax(pred, dim=1), gt)
    
    def train(self, epochs):
        for i in range(1, epochs+1):
            epoch = self.epoch_base + i
            self.model.train()
            if distributed():
                self.dataloader.sampler.set_epoch(epoch)
            
            pbar = tqdm(
                self.dataloader,
                desc=f'Epoch [{epoch}/{epochs + self.epoch_base}]',
                disable=not rank_0(),
                leave=False
            )

            for batch_idx, x in enumerate(pbar):
                x = x.to(self.device)
                labels = x.clone()
                self.optimizer.zero_grad()

                # with autocast("cuda"): ## for new torch version
                with autocast(): ## for old torch version
                    outputs, mus, logvars = self.model(x)
                    ce_loss = self.ce_loss(outputs, labels)
                    kl_loss = self.kl_loss(mus, logvars)
                    lovasz_loss = self.lovasz_loss(outputs, labels)
                    loss = ce_loss + self.kl_coeff * kl_loss + lovasz_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update() 

                # loss.backward()
                # self.optimizer.step()

                if rank_0():
                    pbar.set_postfix({'Step': f"[{batch_idx+1}/{len(self.dataloader)}]",'Loss': f'{loss.item():.4f}', 'KL Loss': f'{kl_loss.item():.4f}', 'Lovasz Loss': f'{lovasz_loss.item():.4f}'})
            
            if epoch % 5 == 0:
                self.save_model(f'vae_epoch_{epoch}.pth')
                self.evaluation()
            self.save_model('latest_vae.pth')
    
    def evaluation(self):
        # print("**************Evaluating****************")
        self.model.eval()
        eval_dataloader, num_classes, _ = get_OccNuscenes_dataloader(data_path = self.data_path, cfg_path = self.cfg_path, batch_size=self.batch_size, train=False)
        miou_metric = Metric_mIoU(
            num_classes=num_classes,
            use_lidar_mask=False,
            use_image_mask=False,
            logger=None
        )
        pbar = tqdm(eval_dataloader, disable=not rank_0())
        for batch_idx, x in enumerate(pbar):
            x = x.to(self.device)
            labels = x.clone()

            # with torch.no_grad(), autocast("cuda"): #for new torch version
            with torch.no_grad(), autocast(): #for old torch version
                outputs = self.model.module.encode_decode(x)

            miou_metric.add_batch(outputs, labels)
            miou_metric.add_iou_batch(outputs, labels)

        dist.barrier() #important to do this
        gathered_hist = [torch.zeros_like(miou_metric.hist) for _ in range(dist.get_world_size())]
        gathered_iou_hist = [torch.zeros_like(miou_metric.iou_hist) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_hist, miou_metric.hist)
        dist.all_gather(gathered_iou_hist, miou_metric.iou_hist)

        if rank_0():
            miou_metric.hist = torch.stack(gathered_hist).sum(0)
            miou_metric.iou_hist = torch.stack(gathered_iou_hist).sum(0)

            _, miou = miou_metric.count_miou()
            iou = miou_metric.count_iou()
            eval_dict = {
                'semantics_miou': miou.item(),
                'binary_iou': iou.item()
            }
            print(eval_dict)


    def load_from_pretrained(self, path, epoch):
        if distributed():
            self.model.module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        else:
            self.model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        self.epoch_base = epoch
        
    @func_rank_0
    def save_model(self, path):
        torch.save(self.model.module.state_dict() if distributed() else self.model.state_dict(), path)
