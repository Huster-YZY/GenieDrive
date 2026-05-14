######################Performance of our vae_d64_210.pth######################
#        [0.8154, 0.9376, 0.9440, 0.8760, 0.8780, 0.7749, 0.9392, 0.9361, 0.9373,
#         0.8277, 0.8630, 0.9415, 0.9601, 0.9255, 0.8626, 0.6505, 0.5771, 0.9854]
# {'semantics_miou': 86.1552734375, 'binary_iou': 75.5275650024414}
##############################################################################
import argparse
import warnings

from utils.dist_utils import *
from trainer import VAETrainer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data."
    )

    parser.add_argument(
        "--cfg_path",
        type=str,
        required=True,
        help="Path to config file."
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to checkpoint file."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    warnings.filterwarnings("ignore")

    set_tf32(True)
    rank, device = setup_dist(verbose=True)

    trainer = VAETrainer(
        device=device,
        data_path=args.data_path,
        cfg_path=args.cfg_path,
        dim_latent=64,
        batch_size=8,
        use_focal_loss=False,
        dropout_rate=0.5,
    )

    trainer.load_from_pretrained(args.ckpt_path, epoch=210)
    trainer.evaluation()

    cleanup_dist()


if __name__ == "__main__":
    main()