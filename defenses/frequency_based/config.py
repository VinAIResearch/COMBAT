import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    # Directory option
    parser.add_argument("--data_root", type=str, default="../../data/")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="original")
    parser.add_argument("--input_height", type=int, default=32)
    parser.add_argument("--input_width", type=int, default=32)
    parser.add_argument("--input_channel", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--attack_mode", type=str, default="all2one")
    parser.add_argument("--n_iters", type=int, default=50)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--noise_rate", type=float, default=0.08)
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--load_checkpoint", default="../../checkpoints")
    parser.add_argument("--saving_prefix", type=str, help="Folder in /checkpoints for saving ckpt")
    parser.add_argument("--scale_noise_rate", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true", default=False)

    return parser
