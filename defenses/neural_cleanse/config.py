import argparse


def get_argument():
    parser = argparse.ArgumentParser()

    # Directory option
    parser.add_argument("--checkpoints", type=str, default="../../checkpoints/")
    parser.add_argument("--data_root", type=str, default="../../data/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--result", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--attack_mode", type=str, default="all2one")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--saving_prefix", type=str, help="Folder in /checkpoints for saving ckpt")

    # ---------------------------- For Neural Cleanse --------------------------
    # Model hyperparameters
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)
    parser.add_argument("--init_cost", type=float, default=1e-3)
    parser.add_argument("--atk_succ_threshold", type=float, default=99.0)
    parser.add_argument("--early_stop", type=bool, default=True)
    parser.add_argument("--early_stop_threshold", type=float, default=99.0)
    parser.add_argument("--early_stop_patience", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cost_multiplier", type=float, default=2)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--target_label", type=int)
    parser.add_argument("--total_label", type=int)
    parser.add_argument("--EPSILON", type=float, default=1e-7)

    parser.add_argument("--to_file", type=bool, default=True)
    parser.add_argument("--n_times_test", type=int, default=1)

    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--S2", type=int, default=8)  # low-res grid size
    parser.add_argument("--grid-rescale", type=float, default=1)  # scale grid values to avoid going out of [-1, 1]. For example, grid-rescale = 0.98
    parser.add_argument("--clamp", action="store_true")  # clamp grid values to [-1, 1]
    parser.add_argument("--nearest", type=int, default=0)  # control grid round-up precision
    #     0: No round-up, just use interpolated input values   (smooth, blur)
    #     1: Round-up to pixel precision                       (sharp, noisy)
    #     2: Round-up to 1/2 pixel precision                   (moderate)

    parser.add_argument("--lnoise", type=int, default=8)  # Length of the input noise vector in dynamic mode
    return parser
