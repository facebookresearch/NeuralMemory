import os
import argparse
import uuid
from pathlib import Path
import submitit
from main import get_parser, main
import copy
import time
import random


class SubmititMain:
    def __init__(self, func):
        self.func = func

    def __call__(self, args):
        self.func(args)

    def checkpoint(self, args):
        print("LOADING CHECKPOINT")
        args.overwrite_checkpoint = False  # load from previous checkpoint
        # bit hacky, but remove dist init file
        try:
            os.remove(args.dist_init[7:])
        except:
            print("ERROR removing dist init file")
        return submitit.helpers.DelayedSubmission(self, args)


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="", help="")
parser.add_argument("--folder", type=str, default="", help="")
parser.add_argument("--partition", type=str, default="learnfair", help="")
parser.add_argument("--ngpus", type=int, default=8, help="")
parser.add_argument("--nodes", type=int, default=1, help="")
parser.add_argument("--mem_per_cpu", type=int, default=2048, help="")
parser.add_argument("--constraint", type=str, default="volta32gb", help="")
parser.add_argument("--time", type=int, default=72, help="hours")
parser.add_argument("--args", type=str, default="", help="")
parser.add_argument("--num_seeds", type=int, default=5, help="")
parser.add_argument("--run_type", type=str, default="", help="")
args = parser.parse_args()

job_parser = get_parser()

seed_start = 0
# seed_end = 3
if args.run_type in ["gpt", "gpt_medium"]:
    random_searches = 1
    enc_layers = [1]
    embedding_dims = [768]
    lrs = [0.00005]
    batch_szs = [64]
    # lrs = [0.0001]
    # batch_szs = [4]
    seeds = range(seed_start, seed_start + 1)
elif args.run_type == "small":
    random_searches = 1
    enc_layers = [4]
    embedding_dims = [256]
    lrs = [0.0001]
    batch_szs = [64]
    seeds = range(seed_start, seed_start + args.num_seeds)
else:
    random_searches = 10
    enc_layers = [4]
    embedding_dims = [256]
    lrs = [0.00005]
    batch_szs = [64]
    seeds = range(seed_start, seed_start + args.num_seeds)

jobs = []
num_jobs = 0
# for i in range(random_searches):
## RANDOM SEARCH ##
# enc_layer = random.choice(enc_layers)
# embedding_dim = random.choice(embedding_dims)
# lr = random.choice(lrs)
# batch_sz = random.choice(batch_szs)

## GRID SEARCH ##
for enc_layer in enc_layers:
    for embedding_dim in embedding_dims:
        for lr in lrs:
            for batch_sz in batch_szs:
                for seed in seeds:
                    # init run folder name and make dir
                    model_name = "layers_{}.dim_{}.lr_{}.bz_{}.seed_{}".format(
                        enc_layer, embedding_dim, lr, batch_sz, seed
                    )
                    folder_str = os.path.join(args.folder, model_name)
                    folder = Path(folder_str)
                    os.makedirs(folder_str, exist_ok=True)

                    # update args
                    job_args = args.args.split()
                    init_file = (
                        folder / f"{uuid.uuid4().hex}_init"
                    )  # not used when nodes=1
                    job_args += ["--submitit", "--dist-init", init_file.as_uri()]
                    job_args = job_parser.parse_args(job_args)
                    job_args_copy = copy.deepcopy(job_args)
                    job_args_copy.seed = seed
                    job_args_copy.enc_layers = enc_layer
                    job_args_copy.embedding_dim = embedding_dim
                    job_args_copy.dim_feedforward = embedding_dim * 2
                    job_args_copy.lr = lr
                    job_args_copy.batch_sz = batch_sz
                    # job_args_copy.checkpoint_dir = folder_str #args.folder

                    # init submitit
                    job_name = args.name + "." + model_name
                    executor = submitit.AutoExecutor(
                        folder=folder, slurm_max_num_timeout=10
                    )
                    executor.update_parameters(name=job_name)
                    executor.update_parameters(
                        mem_gb=256,
                        gpus_per_node=args.ngpus,
                        tasks_per_node=args.ngpus,  # one task per GPU
                        cpus_per_task=10,
                        nodes=args.nodes,
                        timeout_min=args.time * 60,
                        slurm_partition=args.partition,
                        slurm_signal_delay_s=120,
                        slurm_additional_parameters={
                            "mail-user": f"{os.environ['USER']}@fb.com",
                            "mail-type": "fail",
                        },
                    )
                    submitit_main = SubmititMain(main)
                    job = executor.submit(submitit_main, job_args_copy)
                    print("submited {} {}".format(job.job_id, job_name))
                    num_jobs += 1

print("\nJOBS: {}".format(num_jobs))
