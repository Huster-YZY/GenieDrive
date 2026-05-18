import torch


src = ""
vae = "ckpts/vae_d64_210.pth"

def f(src, vae_path, dst=None):
    ar = torch.load(src, map_location="cpu")
    vae = torch.load(vae_path, map_location="cpu")

    new_vae_dict = {f'vqvae.{key}': value for key, value in vae.items()}

    ar['state_dict'].update(new_vae_dict)
    dst_path = dst if dst is not None else "work_dirs/ckpt_stage2_init.pth"
    torch.save(ar, dst_path)

f(src, vae)
print("done")