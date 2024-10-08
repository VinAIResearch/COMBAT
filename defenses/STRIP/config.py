import argparse


def get_argument():
    parser = argparse.ArgumentParser()

    # Directory option
    parser.add_argument("--data_root", type=str, default="../../data/")
    parser.add_argument("--checkpoints", type=str, default="../../checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--attack_mode", type=str, default="all2one")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--noise_rate", type=float, default=0.08)
    parser.add_argument("--ratio", type=float, default=0.65, help="scale ratio for DCT of noise")
    parser.add_argument("--kernel_size", type=int, default=3, help="kernel size for Gaussian blur")
    parser.add_argument("--sigma", type=tuple, default=(0.1, 1.0), help="sigma for Gaussian blur")

    # ---------------------------- For STRIP --------------------------
    # Model hyperparameters
    parser.add_argument("--n_sample", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--detection_boundary", type=float, default=0.2)  # According to the original paper
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--test_rounds", type=int, default=10)

    return parser
