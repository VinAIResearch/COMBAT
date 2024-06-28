import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../../data/")
    parser.add_argument("--checkpoints", type=str, default="../../checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--saving_prefix", type=str,
                        help="Folder in /checkpoints for saving ckpt")

    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--noise_rate", type=float, default=0.08)
    parser.add_argument("--ratio", type=float, default=0.65,
                        help="scale ratio for DCT of noise")
    parser.add_argument("--kernel_size", type=int, default=3,
                        help="kernel size for Gaussian blur")
    parser.add_argument("--sigma", type=tuple, default=(0.1,
                        1.0), help="sigma for Gaussian blur")

    parser.add_argument("--bs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--attack_mode", type=str,
                        default="all2one", help="all2one or all2all")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--outfile", type=str, default="./results.txt")

    parser.add_argument("--S2", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument(
        "--grid-rescale", type=float, default=1
    )  # scale grid values to avoid going out of [-1, 1]. For example, grid-rescale = 0.98
    # clamp grid values to [-1, 1]
    parser.add_argument("--clamp", action="store_true")
    # control grid round-up precision
    parser.add_argument("--nearest", type=float, default=0)
    #     0: No round-up, just use interpolated input values   (smooth, blur)
    #     1: Round-up to pixel precision                       (sharp, noisy)
    #     2: Round-up to 1/2 pixel precision                   (moderate)
    return parser
