import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--saving_prefix", type=str, help="Folder in /checkpoints for saving ckpt")
    parser.add_argument("--attack_mode", default="all2one")
    parser.add_argument("--load_checkpoint", default="")
    parser.add_argument("--load_checkpoint_clean", type=str)

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--input_height", type=int, default=32)
    parser.add_argument("--input_width", type=int, default=32)
    parser.add_argument("--input_channel", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr_C", type=float, default=1e-2)
    parser.add_argument("--lr_G", type=float, default=1e-2)
    parser.add_argument("--lr_clean", type=float, default=1e-2)
    parser.add_argument("--schedulerC_milestones", type=list, default=[100, 150])
    parser.add_argument("--schedulerG_milestones", type=list, default=[100, 150])
    parser.add_argument("--scheduler_clean_milestones", type=list, default=[100, 150])
    parser.add_argument("--schedulerC_lambda", type=float, default=0.1)
    parser.add_argument("--schedulerG_lambda", type=float, default=0.1)
    parser.add_argument("--scheduler_clean_lambda", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lambda_cov", type=float, default=1)

    parser.add_argument("--noise_rate", type=float, default=0.08)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--pc", type=float, default=0.5)
    parser.add_argument("--cross_rate", type=float, default=1)
    parser.add_argument("--s", type=int, default=2)
    parser.add_argument("--grid_rescale", type=float, default=0.15)
    parser.add_argument("--ratio", type=float, default=0.65, help="scale ratio for DCT of noise")
    parser.add_argument("--kernel_size", type=int, default=3, help="kernel size for Gaussian blur")
    parser.add_argument("--sigma", type=tuple, default=(0.1, 1.0), help="sigma for Gaussian blur")

    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)

    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--S2", type=int, default=8)  # low-res grid size
    parser.add_argument("--clamp", action="store_true")  # clamp grid values to [-1, 1]
    parser.add_argument("--nearest", type=float, default=0)  # control grid round-up precision
    #     0: No round-up, just use interpolated input values   (smooth, blur)
    #     1: Round-up to pixel precision                       (sharp, noisy)
    #     2: Round-up to 1/2 pixel precision                   (moderate)

    parser.add_argument("--lnoise", type=int, default=8)  # Length of the input noise vector in dynamic mode

    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--tv_weight", type=float, default=0.01)
    parser.add_argument("--L2_weight", type=float, default=0.02)
    parser.add_argument("--F_checkpoints", type=str, default="./defenses/frequency_based/checkpoints")
    parser.add_argument("--F_model", type=str, default="original")
    parser.add_argument("--F_dropout", type=float, default=0.5)
    parser.add_argument("--F_num_ensemble", type=int, default=3)

    parser.add_argument("--model_clean", type=str, default="default")
    parser.add_argument("--clean_model_weight", type=float, default=0.8) # weight of clean model loss

    parser.add_argument("--noise_only", action="store_true", default=False)
    parser.add_argument("--post_transform_option", type=str, default="use", choices=["use", "no_use", "use_modified"])
    parser.add_argument("--scale_noise_rate", type=float, default=1.0)

    parser.add_argument("--cross_weight", type=float, default=0.2)

    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument("--r", type=float, default=1/4)
    parser.add_argument("--scale_factor", type=float, default=0.5)
    parser.add_argument("--scale_mode", type=str, default="bicubic")

    return parser
