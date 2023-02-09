# Run this script from SuperPOD login node only

import argparse
import json
import subprocess


def get_from_squeue(args):
    result = subprocess.run(["squeue", "--me", "--sort", "i", "-ho", "%j %i"], stdout=subprocess.PIPE)
    result = result.stdout.decode("utf-8")
    lines = result.strip().split("\n")

    return {line.split()[0]: int(line.split()[1]) for line in lines}


def get_from_sacct(args):
    result = subprocess.run("sacct -o jobname,jobid -nXP".split(), stdout=subprocess.PIPE)
    result = result.stdout.decode("utf-8")
    lines = result.strip().split("\n")

    return {line.split("|")[0]: int(line.split("|")[1]) for line in lines}


def main(args):
    if args.source == "squeue":
        jobname2id = get_from_squeue(args)
    if args.source == "sacct":
        jobname2id = get_from_sacct(args)
    print(json.dumps(jobname2id, indent=4))
    json.dump(jobname2id, open(args.output_file, "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="squeue")
    parser.add_argument("--output_file", type=str, default="jobname2id.json")
    args = parser.parse_args()

    main(args)
