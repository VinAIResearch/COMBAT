import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../../data/")
    parser.add_argument("--checkpoints", type=str,
                        default="../../checkpoints/")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--saving_prefix", type=str,
                        help="Folder in /checkpoints for saving ckpt")
    parser.add_argument("--load_checkpoint_clean", type=str)
    parser.add_argument("--results", type=str, default="./results")

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--input_height", type=int, default=32)
    parser.add_argument("--input_width", type=int, default=32)
    parser.add_argument("--input_channel", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--noise_rate", type=float, default=0.08)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.65,
                        help="scale ratio for DCT of noise")
    parser.add_argument("--kernel_size", type=int, default=3,
                        help="kernel size for Gaussian blur")
    parser.add_argument("--sigma", type=tuple, default=(0.1,
                        1.0), help="sigma for Gaussian blur")

    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)
    parser.add_argument("--attack_mode", type=str,
                        default="all2one", help="all2one or all2all")

    return parser
