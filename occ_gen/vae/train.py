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
        help="Path to training data."
    )

    parser.add_argument(
        "--cfg_path",
        type=str,
        required=True,
        help="Path to config file."
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

    trainer.train(epochs=210)
    trainer.evaluation()

    cleanup_dist()


if __name__ == "__main__":
    main()