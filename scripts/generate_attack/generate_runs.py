import argparse
import json
import os
import re
from collections import defaultdict
from itertools import product
from glob import glob
from copy import deepcopy

import numpy as np

base_cmd_template = "python {task}_{attack}.py --dataset {dataset} --saving_prefix {saving_prefix}"
base_fd_cmd_template = "python {task}.py --dataset {dataset} --model {model}"
base_eval_img_sim_cmd_template = "python evaluate_img_sim.py --dataset {dataset} --model {model}"

jobname2id = json.load(open("jobname2id.json"))


def run_fd(args, f, dataset, saving_prefix):
    template = open(args.template_path, "r").read()
    #pwd = "/root/CleanLabelBackdoorGenerator/scripts/generate_attack"
    pwd = "/root/CleanLabelBackdoorGenerator/defenses/frequency_based"

    for FD_task in args.FD_tasks:
        if FD_task == "test":
            f.write(f"echo TESTING CHECKPOINT: {saving_prefix} " + "-"*50 + "\n")
        for FD_model in args.FD_models:
            cmd_template = base_fd_cmd_template

            job_name = f"{FD_task}_frequency_defense_{dataset}_{FD_model}"

            if FD_task == "test":
                noise_rate = float(re.search("(?<=noise_rate)\d*.?\d*", saving_prefix)[0])
                snr = re.search("(?<=snr)\d*.?\d*", saving_prefix)
                if snr:
                    snr = float(snr[0])
                    noise_rate *= snr
                cmd_template += f" --noise_rate {noise_rate}"
                job_name += f"_saving_prefix{saving_prefix}"
                cmd_template += f" --saving_prefix {saving_prefix}"

            cmd = cmd_template.format(task=FD_task, dataset=dataset, model=FD_model)

            slurm_config = args.slurm_config[:]
            slurm_config = "\n".join(slurm_config)

            g = open(job_name + ".sh", "w")
            g.write(
                template.format(
                    partition=args.partition,
                    slurm_log=args.slurm_log,
                    workspace=args.workspace,
                    cpu_per_task=args.cpu_per_task,
                    mem=args.mem,
                    env=args.env,
                    job_name=job_name,
                    cmd=cmd,
                    slurm_config=slurm_config,
                )
            )
            #g.write(cmd + "\n")
            g.close()

            f.write(f"sbatch {job_name}.sh\n")
            #f.write(f"bash {pwd}/{job_name}.sh\n")
            #f.write(f"echo {cmd}\n")
            #f.write(f"{cmd}\n\n")
    f.write("\n")
    #f.write("echo\necho\necho\n")


def make_saving_prefix(task, params, add_clean_postfix=False, add_holdout_postfix=False, level=1):
    """From a set of params, construct a unique saving prefix for that setting"""
    def check_skip_conditions(name, value):
        return (name == "model" and value == "default") or \
            (name == "model_clean" and value == "default") or \
            (name == "F_model" and value == "original") or \
            (name == "F_model_eval" and value == "original_holdout") or \
            (name == "target_label" and value == 0) or \
            name.startswith("load_checkpoint")

    # def check_skip_conditions_2(name, value):
    #     # return (name == "model" and value == "default") or \
    #     #     (name == "model_clean" and value == "default") or \
    #     return (name == "F_model" and value == "original") or \
    #         (name == "F_model_eval" and value == "original_holdout") or \
    #         (name == "target_label" and value == 0) or \
    #         name.startswith("load_checkpoint")

    # def shorten_name(name):
    #     if name == "clean_model_weight":
    #         name = "cweight"
    #     if name == "post_transform_option":
    #         name = "ptopt"
    #     if name == "scale_noise_rate":
    #         name = "snr"
    #     return name

    saving_prefix = [task]

    if task.startswith("train_generator"):
        for name, value in params.items():
            # Skip these args to reduce prefix length
            if check_skip_conditions(name, value):
            # if check_skip_conditions_2(name, value):
                continue
            if name == "dataset":
                name = ""
            # name = shorten_name(name)
            if value is None:
                value = ""
            saving_prefix += [f"{name}{value}"]

    if task.startswith("train_victim"):
        for name, value in params.items():
            # Skip these args to reduce prefix length
            if check_skip_conditions(name, value):
            # if check_skip_conditions_2(name, value):
                continue
            if name == "dataset":
                name = ""
            # name = shorten_name(name)
            if value is None:
                value = ""
            saving_prefix += [f"{name}{value}"]

    if task.startswith("train_classifier_only"):
        for name, value in params.items():
            # Skip these args to reduce prefix length
            if check_skip_conditions(name, value):
                continue
            if name == "dataset":
                name = ""
            if value is None:
                value = ""
            saving_prefix += [f"{name}{value}"]

    if task.startswith("eval"):
        for name, value in params.items():
            # Skip these args to reduce prefix length
            if check_skip_conditions(name, value):
                continue
            if name == "dataset":
                name = ""
            saving_prefix += [f"{name}{value}"]

    # Misc args
    if args["at"]:
        saving_prefix += ["at"]
    if args["debug"] and level == 0:
        saving_prefix += ["debug"]

    if add_holdout_postfix:
        saving_prefix += ["holdout"]
    if add_clean_postfix:
        saving_prefix += ["clean"]

    return "_".join(saving_prefix)


def find_train_generator_saving_prefix(task, params, add_clean_postfix=False):
    """Return corresponding train_generator saving prefix given a set of train_victim params"""
    train_params = deepcopy(params)

    # For now, regardless of the value of model and pc in stage 2,
    # we still load checkpoint in stage 1 that uses model=default, pc=0.5
    train_params["model"] = "default"
    # train_params["model"] = "preactresnetdropout18"
    train_params["pc"] = 0.5

    train_saving_prefix = make_saving_prefix(task.replace("train_victim", "train_generator"), train_params, add_clean_postfix)
    return train_saving_prefix


def make_cmd(task, params, level=0):
    cmd = f"python {task}.py"

    if task.startswith("train_generator"):
        for name, value in params.items():
            if name == "load_checkpoint_f":
                name = "load_checkpoint"
                f_params = {arg: params[arg] for arg in task_args["train_classifier_only"]}
                value = make_saving_prefix("train_classifier_only", f_params, True)
            if name == "load_checkpoint_h":
                name = "load_checkpoint_clean"
                h_params = {arg: params[arg] for arg in task_args["train_classifier_only"]}
                # h_params["model"] = "default"
                value = make_saving_prefix("train_classifier_only", h_params, True, True)
            if name == "lr":  # Hack :P
                name = f"lr_C {value}"
                value = f"--lr_G {value}"
            if value is None:
                value = ""
            cmd += f" --{name} {value}"

    if task.startswith("train_victim"):
        for name, value in params.items():
            if name == "load_checkpoint":
                value = find_train_generator_saving_prefix(task, params, True)
            if name == "lr":  # Hack :P
                name = f"lr_C {value}"
                value = f"--lr_G {value}"
            if value is None:
                value = ""
            cmd += f" --{name} {value}"

    if task.startswith("train_classifier_only"):
        for name, value in params.items():
            if name == "lr":  # Hack :P
                name = f"lr_C {value}"
                value = f"--lr_G {value}"
            if value is None:
                value = ""
            cmd += f" --{name} {value}"

    if task.startswith("eval"):
        for name, value in params.items():
            if name == "load_checkpoint_h":
                name = "load_checkpoint_clean"
                C_params = {arg: params[arg] for arg in task_args["train_classifier_only"]}
                value = make_saving_prefix("train_classifier_only", C_params, True, True)
            if name == "load_checkpoint":
                G_params = {arg: params[arg] for arg in task_args["train_victim_fixh"]}  # TODO: Implement more complex behavior later with args.eval_tasks
                value = find_train_generator_saving_prefix("train_victim_fixh", G_params, True)
            cmd += f" --{name} {value}"

    # Misc args
    if args["at"]:
        cmd += f" --at"
    if args["continue_training"]:
        cmd += f" --continue_training"

    saving_prefix = make_saving_prefix(task, params, level=level)
    cmd += f" --saving_prefix {saving_prefix}"

    if args["debug"]:
        cmd += f" --debug"
        cmd += f" --n_iters 2"

    return cmd



def make_FD_saving_prefix(task, params, dataset, add_clean_postfix=False, add_holdout_postfix=False, level=1):
    """From a set of params, construct a unique saving prefix for that setting"""

    saving_prefix = [task]

    if task.startswith("train"):
        for name, value in params.items():
            name = name.replace("FD_", "")
            if name == "dataset":
                name = ""
                value = dataset if value == "same" else value
            saving_prefix += [f"{name}{value}"]

    if task.startswith("test"):
        for name, value in params.items():
            name = name.replace("FD_", "")
            if name == "dataset":
                name = ""
                value = dataset if value == "same" else value
            saving_prefix += [f"{name}{value}"]

    if args["debug"] and level == 0:
        saving_prefix += ["debug"]

    if add_holdout_postfix:
        saving_prefix += ["holdout"]
    if add_clean_postfix:
        saving_prefix += ["clean"]

    return "_".join(saving_prefix)


def make_FD_cmd(task, params, dataset, level=0):
    cmd = f"python {task}.py"

    if task.startswith("train"):
        for name, value in params.items():
            name = name.replace("FD_", "")
            if name == "dataset" and value == "same":
                value = dataset
            cmd += f" --{name} {value}"

    if task.startswith("test"):
        for name, value in params.items():
            name = name.replace("FD_", "")
            if name == "dataset" and value == "same":
                value = dataset
            cmd += f" --{name} {value}"

    if args["debug"]:
        cmd += f" --debug"
        cmd += f" --n_iters 2"

    return cmd


def FD_main(args, FD_task_args, saving_prefix, dataset, slurm_config, f):
    for FD_task in args["FD_tasks"]:
        FD_params_combinations = product(*[args.get(arg, [None]) for arg in FD_task_args[FD_task]])
        for FD_params in FD_params_combinations:
            FD_params = dict(zip(FD_task_args[FD_task], FD_params))
            if "saving_prefix" in FD_params:
                FD_params["saving_prefix"] = saving_prefix

            cmd = make_FD_cmd(FD_task, FD_params, dataset, level=0)
            new_saving_prefix = make_FD_saving_prefix(FD_task, FD_params, dataset, level=0)

            export(args, new_saving_prefix, cmd, slurm_config, f)
            print(f"Created {new_saving_prefix}")


def make_sbatch(args, saving_prefix, cmd, slurm_config):
    template = open(args["template_path"], "r").read()
    return template.format(
        partition=args["partition"],
        slurm_log=args["slurm_log"],
        workspace=args["workspace"],
        cpu_per_task=args["cpu_per_task"],
        mem=args["mem"],
        env=args["env"],
        job_name=saving_prefix,
        cmd=cmd,
        slurm_config='\n'.join(slurm_config),
    )


def export(args, saving_prefix, cmd, slurm_config, f):
    if args["output_format"] == "sbatch":
        sbatch = make_sbatch(args, saving_prefix, cmd, slurm_config)
        with open(saving_prefix + ".sh", "w") as g:
            g.write(sbatch)
        f.write(f"sbatch {saving_prefix}.sh\n")
    elif args["output_format"] == "bash":
        f.write(f"echo cd {args['workspace']}\n")
        f.write(f"cd {args['workspace']}\n")
        f.write(f"echo {cmd}\n{cmd}\n")


def main(args, task_args, FD_task_args, level=0):
    if args["run_fd"]:
        args["workspace"] = "/root/CleanLabelBackdoorGenerator/defenses/frequency_based"
        level=1

    with open(f"run.sh", "w") as f:
        f.write("#!/bin/bash\n\n")

        for task in args["tasks"]:
            params_combinations = product(*[args.get(arg, [None]) for arg in task_args[task]])
            for params in params_combinations:
                params = dict(zip(task_args[task], params))

                slurm_config = args["slurm_config"][:]

                saving_prefix = make_saving_prefix(task, params, level=level)
                cmd = make_cmd(task, params, level=level)

                if args["have_dependency"] and task.startswith("train_victim"):
                    train_saving_prefix = find_train_generator_saving_prefix(task, params)
                    if train_saving_prefix  not in jobname2id:
                        print(f"Cannot find depended job {train_saving_prefix}. Ignore {saving_prefix}")
                        continue
                    slurm_config.append(f"#SBATCH --dependency=afterany:{jobname2id[train_saving_prefix]}")

                if len(saving_prefix) > 255:
                    raise ValueError(f"saving_prefix (len={len(saving_prefix)}) exceed max filename length")

                if args["run_fd"]:
                    FD_main(args, FD_task_args, saving_prefix, params["dataset"], slurm_config, f)
                else:
                    export(args, saving_prefix, cmd, slurm_config, f)
                    print(f"Created {saving_prefix}")

                f.write("\n")
            f.write("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_path", type=str, default="/root/CleanLabelBackdoorGenerator/scripts/generate_attack/template_sbatch.txt")
    parser.add_argument("--partition", type=str, default="research")
    parser.add_argument("--slurm_log", type=str, default="/lustre/scratch/client/vinai/users/dangnm12/CleanLabelBackdoorGenerator/slurm_log")
    parser.add_argument("--workspace", type=str, default="/root/CleanLabelBackdoorGenerator")
    parser.add_argument("--cpu_per_task", type=int, default=16)
    parser.add_argument("--mem", type=int, default=64)
    parser.add_argument("--env", type=str, default="backdoor")
    parser.add_argument("--slurm_config", type=str, nargs="+", default=[])

    # Attack args
    parser.add_argument("--debug", default=False)
    parser.add_argument("--output_format", type=str, choices=["sbatch", "bash"],
        default="sbatch",
        # default="bash",
    )
    parser.add_argument("--have_dependency", default=False)
    parser.add_argument("--continue_training", default=False)

    task_args = {
        "train_classifier_only":                            ["dataset", "model"],
        # "train_classifier_only":                            ["dataset", "model", "num_classes"],

        # eval (load_checkpoint = G's checkpoint, load_checkpoint_h = C's checkpoint)
        "eval":                                             ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_h", "load_checkpoint"],

        "train_generator_targetclassonly":                  ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "model", "F_model", "F_model_eval", "num_classes"],
        "train_generator_targetclassonlywithadv":           ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "model", "F_model", "F_model_eval", "num_classes"],

        # Jointly train h
        "train_generator":                                  ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "F_model", "F_model_eval"],
        "train_generator_fixhimperceptible":                ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "tv_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_h"],
        "train_generator_fixhwanet":                        ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "s", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_h"],
        "train_generator_fixhinputaware":                   ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "cross_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_h"],
        "train_generator_fixhmultilabel":                   ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_h"],

        # Fix h (load_checkpoint_f = load_checkpoint, load_checkpoint_h = load_checkpoint_clean)
        # "train_generator_fixh":                             ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_h"],
        # "train_generator_fixh":                             ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_h", "num_classes"],
        "train_generator_fixh":                             ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_h", "lr_C", "lr_G"],
        # "train_generator_fixh":                             ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_h", "lr", "F_dropout", "post_transform_option", "scale_noise_rate"],
        # "train_generator_fixh":                             ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_h", "F_dropout", "post_transform_option", "scale_noise_rate"],
        # "train_generator_fixh":                             ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_h", "new_normalize"],
        # "train_generator_fixh":                             ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_h", "new_normalize", "F_dropout", "post_transform_option", "scale_noise_rate"],
        "train_generator_fixhdropout":                      ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_h"],
        "train_generator_fixhscheduledcleanmodelweight":    ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_h"],
        # "train_generator_fixfh":                            ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_f", "load_checkpoint_h"],
        "train_generator_fixfh":                            ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint_f", "load_checkpoint_h", "lr_C", "lr_G", "lr_clean"],
        "train_generator_fixhinterpolate":                  ["dataset", "noise_rate", "pc", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "load_checkpoint_h", "scale_factor", "scale_mode"],

        # Low-frequency trigger
        "train_generator_lowfreq":                          ["dataset", "noise_rate", "pc", "L2_weight", "model", "F_model", "r"],




        "train_victim_targetclassonly":                     ["dataset", "noise_rate", "pc", "F_weight", "L2_weight",  "model", "F_model", "F_model_eval", "load_checkpoint", "num_classes"],
        "train_victim_targetclassonlywithadv":              ["dataset", "noise_rate", "pc", "F_weight", "L2_weight",  "model", "F_model", "F_model_eval", "load_checkpoint", "num_classes"],

        # Jointly train h
        "train_victim":                                     ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "F_model", "F_model_eval", "load_checkpoint"],
        "train_victim_fixhimperceptible":                   ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "tv_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint"],
        "train_victim_fixhwanet":                           ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "s", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint"],
        "train_victim_fixhinputaware":                      ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "cross_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint"],
        "train_victim_fixhmultilabel":                      ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint"],

        # Fix h
        # "train_victim_fixh":                                ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint"],
        # "train_victim_fixh":                                ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint", "num_classes"],
        "train_victim_fixh":                                ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint", "lr_C", "lr_G"],
        # "train_victim_fixh":                                ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint", "lr", "F_dropout", "post_transform_option", "scale_noise_rate"],
        # "train_victim_fixh":                                ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint", "F_dropout", "post_transform_option", "scale_noise_rate"],
        # "train_victim_fixh":                                ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint", "new_normalize"],
        # "train_victim_fixh":                                ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint", "new_normalize", "F_dropout", "post_transform_option", "scale_noise_rate"],
        "train_victim_fixhdropout":                         ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint"],
        "train_victim_fixhscheduledcleanmodelweight":       ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint"],
        # "train_victim_fixfh":                               ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint"],
        "train_victim_fixfh":                               ["dataset", "noise_rate", "pc", "F_weight", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "F_model_eval", "load_checkpoint", "lr_C", "lr_G", "lr_clean"],
        "train_victim_fixhinterpolate":                     ["dataset", "noise_rate", "pc", "L2_weight", "clean_model_weight", "model", "model_clean", "F_model", "load_checkpoint", "scale_factor", "scale_mode"],

        # Low-frequency trigger
        "train_victim_lowfreq":                             ["dataset", "noise_rate", "pc", "L2_weight", "model", "F_model", "r", "load_checkpoint"],
    }

    FD_task_args = {
        "train":                                            ["FD_dataset", "FD_model"],
        "test":                                             ["FD_dataset", "FD_model", "saving_prefix"],
    }

    parser.add_argument("--tasks", type=str, nargs="+", default=[
        # "train_classifier_only",
        # "eval",
        # "train_generator_targetclassonly",
        # "train_generator_targetclassonlywithadv",
        # "train_generator",
        # "train_generator_fixhimperceptible",
        # "train_generator_fixhwanet",
        # "train_generator_fixhinputaware",
        # "train_generator_fixhmultilabel",
        # "train_generator_fixh",
        # "train_generator_fixhdropout",
        # "train_generator_fixhscheduledcleanmodelweight",
        # "train_generator_fixfh",
        # "train_generator_fixhinterpolate",
        # "train_generator_lowfreq",

        # "train_victim_targetclassonly",
        # "train_victim_targetclassonlywithadv",
        # "train_victim",
        # "train_victim_fixhimperceptible",
        # "train_victim_fixhwanet",
        # "train_victim_fixhinputaware",
        # "train_victim_fixhmultilabel",
        # "train_victim_fixh",
        # "train_victim_fixhdropout",
        # "train_victim_fixhscheduledcleanmodelweight",
        # "train_victim_fixfh",
        # "train_victim_fixhinterpolate",
        "train_victim_lowfreq",
    ])

    # Args for universal hyperparameters
    parser.add_argument("--dataset", type=str, nargs="+", default=[
        "cifar10",
        # "gtsrb",
        # "celeba",
        # "imagenet10",
        # "imagenet10small",
    ])
    parser.add_argument("--noise_rate", type=float, nargs="+", default=[
        # 0.01,
        # 0.02,
        # 0.03,
        # 0.04,
        # 0.05,
        # 0.06,
        # 0.08,  # CIFAR10, CelebA, ImageNet10
        0.125,
        0.15,  # GTSRB
        0.2,
    ])
    parser.add_argument("--pc", type=float, nargs="+", default=[
        # 0,
        # 0.1,
        # 0.2,
        # 0.3,
        # 0.4,
        0.5,
        # 0.005,
    ])

    # Misc hyperparameters
    parser.add_argument("--lr", type=float, nargs="+", default=[0.001])
    parser.add_argument("--lr_C", type=float, nargs="+", default=[0.001])
    parser.add_argument("--lr_G", type=float, nargs="+", default=[0.001])
    parser.add_argument("--lr_clean", type=float, nargs="+", default=[0.001])
    parser.add_argument("--F_weight", type=float, nargs="+", default=[
        0.08,
        # 0.1,
        # 0.15,
        # 0.2,
    ])
    parser.add_argument("--L2_weight", type=float, nargs="+", default=[0.02])
    parser.add_argument("--clean_model_weight", type=float, nargs="+", default=[0.8])
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--at", type=bool, default=False)
    parser.add_argument("--num_classes", type=int, nargs="+", default=[
        # 10,
        # 13,
        43,
    ])
    parser.add_argument("--r", type=float, nargs="+", default=[
        1/3,
        1/4,
        # 1/8,
        # 1/16,
    ])
    parser.add_argument("--scale_factor", type=float, nargs="+", default=[
        0.5,
    ])
    parser.add_argument("--scale_mode", type=str, nargs="+", default=[
        "bicubic"
    ])

    # Args for many models involve
    parser.add_argument("--model", type=str, nargs="+", default=[
        "default",
        # "preactresnet10",
        # "preactresnetdropout18",
        # "efficientnetb0",
        # "vittiny",
        # "vitsmall",
        # "vitbase"
        # "mobilenetv2",
        # "vgg13",
        # "simplevitsmall8"
    ])
    parser.add_argument("--model_clean", type=str, nargs="+", default=[
        "default",
        # "preactresnet10",
    ])
    parser.add_argument("--F_model", type=str, nargs="+", default=[
        "original",  # CIFAR10, CelebA
        # "original_dropout",  # GTSRB
    ])
    parser.add_argument("--F_model_eval", type=str, nargs="+", default=["original_holdout"])

    # Args specifically for GTSRB
    # parser.add_argument("--F_models", type=list, nargs="+", default=[["original", "mobilenetv2"]])
    # parser.add_argument("--F_num_ensemble", type=int, nargs="+", default=[3])
    parser.add_argument("--F_dropout", type=float, nargs="+", default=[0.5])
    parser.add_argument("--post_transform_option", type=str, nargs="+", choices=["use", "no_use", "use_modified"], default=["no_use"])
    parser.add_argument("--scale_noise_rate", type=float, nargs="+", default=[2.0])

    # Args specifically for customized attacks (imperceptible, inputaware, wanet)
    parser.add_argument("--tv_weight", type=float, nargs="+", default=[
        # 0.0002,
        # 0.001,
        # 0.002,
        0.01,
    ])
    parser.add_argument("--cross_weight", type=float, nargs="+", default=[0.2])
    parser.add_argument("--s", type=int, nargs="+", default=[4])



    # Eval args
    parser.add_argument("--eval_tasks", type=str, nargs="+", default=[
        "train_generator_fixh",
    ])



    # Frequency Defense args
    parser.add_argument("--run_fd", default=False)
    parser.add_argument("--FD_tasks", type=str, nargs="+", default=[
        "train",
        # "test",
    ])

    parser.add_argument("--FD_dataset", type=str, nargs="+", default=[
        "same", # use the same FD checkpoint's dataset with the target model's dataset
        # "cifar10",
        # "gtsrb",
        # "celeba",
        # "imagenet10",
    ])

    parser.add_argument("--FD_model", type=str, nargs="+", default=[
        "original",
        "original_holdout",
        # "densenet121",
        # "vgg13",
        # "mobilenetv2",
        # "resnet18",
        # "efficientnetb0",
        # "googlenet",
        # "googlenetadadelta",
    ])

    args = vars(parser.parse_args())

    main(args, task_args, FD_task_args)
