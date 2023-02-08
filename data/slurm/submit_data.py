import os
import argparse
import uuid
from pathlib import Path
import submitit
from generate_data import main
from config_args import get_opts, read_configs
import copy


class SubmititMain:
    def __init__(self, func):
        self.func = func

    def __call__(self, args):
        self.func(args)

    def checkpoint(self, args):
        return submitit.helpers.DelayedSubmission(self, args)


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="", help="")
parser.add_argument("--folder", type=str, default="", help="")
parser.add_argument("--partition", type=str, default="learnfair", help="")
parser.add_argument("--ngpus", type=int, default=8, help="")
parser.add_argument("--nodes", type=int, default=1, help="")
parser.add_argument("--mem_per_cpu", type=int, default=2048, help="")
parser.add_argument("--constraint", type=str, default="volta32gb", help="")
parser.add_argument("--time", type=int, default=10, help="hours")
parser.add_argument("--slurm_array", action="store_true", default=False)
parser.add_argument("--array_iterations", type=int, default=101, help="")
parser.add_argument("--args", type=str, default="", help="")
args = parser.parse_args()

job_args = args.args.split()
folder = Path(args.folder)
os.makedirs(str(folder), exist_ok=True)
job_parser = get_opts()
job_args = job_parser.parse_args(job_args)
job_args.configs = read_configs(job_args.config_file)

executor = submitit.AutoExecutor(folder=folder / "%j", slurm_max_num_timeout=10)
executor.update_parameters(
    mem_gb=128,
    gpus_per_node=0,
    tasks_per_node=1,  # one task per node
    cpus_per_task=2,
    nodes=1,
    timeout_min=args.time * 60,
    slurm_partition=args.partition,
    slurm_signal_delay_s=120,
    slurm_additional_parameters={
        "mail-user": f"{os.environ['USER']}@fb.com",
        "mail-type": "fail",
    },
)

executor.update_parameters(name=args.name)
executor.update_parameters(slurm_array_parallelism=1024)
submitit_main = SubmititMain(main)

start_seed = job_args.seed

if args.slurm_array:
    seeds = range(start_seed, start_seed + args.array_iterations)
    # seeds = range(319,575)
    jobs = []
    with executor.batch():
        for seed in seeds:
            job_args_seed = copy.deepcopy(job_args)
            job_args_seed.seed = seed
            job = executor.submit(submitit_main, job_args_seed)
            jobs.append(job)
else:
    job = executor.submit(submitit_main, job_args)
    print("submited {} {}".format(job.job_id, args.name))
