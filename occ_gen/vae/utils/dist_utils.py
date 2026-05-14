import os
import torch

import torch.distributed as dist


def distributed():
    return dist.is_initialized()


def rank_0():
    return not distributed() or dist.get_rank() == 0


def func_rank_0(func):
    return lambda *args, **kwargs: func(*args, **kwargs) if rank_0() else None


@func_rank_0
def write_text(text):
    print(text, end='')


@func_rank_0
def flush_text():
    print('\r', end='')


@func_rank_0
def print_text(*args, **kwargs):
    print(*args, **kwargs)


@func_rank_0
def wlog(*args, **kwargs):
    import wandb
    wandb.log(*args, **kwargs)

def setup_dist(verbose=True):
    """
    Setup distributed training if script is launched with torchrun.
    Currently only supports single node multi GPU.
    """
    if all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']):
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        if verbose:
            print(f'Starting rank={rank}, world_size={dist.get_world_size()}.')
    else:
        rank = device = 0
    return rank, device

def set_tf32(use_tf32=True):
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

def cleanup_dist():
    if distributed():
        dist.destroy_process_group()