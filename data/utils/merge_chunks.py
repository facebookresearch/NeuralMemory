import argparse
import os
import torch
import re


def merge(data_path, new_data_path, new_shard_size):
    """
    This function merges small data shards into bigger data shards,
    where the bigger shard size is defined by args.new_shard_size

    Input: path to directory where small shards are located
    Output: writes new files in the format train_1_2.pth, train_2_2.pth, train_3_2.pth,
    """

    new_shard = []
    shard_idx = 0
    for filename in os.listdir(data_path):
        if "train_" in filename:
            f = os.path.join(data_path, filename)
            if os.path.isfile(f):
                shard = torch.load(f)
                if len(shard) < new_shard_size:
                    new_shard += shard
                    if len(new_shard) >= new_shard_size:
                        if shard_idx == 0:
                            new_shard_file = os.path.join(new_data_path, "test.pth")
                        elif shard_idx == 1:
                            new_shard_file = os.path.join(new_data_path, "val.pth")
                        else:
                            new_shard_file = os.path.join(
                                new_data_path, "train_{}_1.pth".format(shard_idx)
                            )
                        print("saving shard {}".format(new_shard_file))
                        torch.save(new_shard, new_shard_file)
                        shard_idx += 1
                        new_shard = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    A = parser.add_argument
    A("--new_shard_size", type=int, default=10000)
    A("--data_root", type=str, default="data/data/")
    A(
        "--dataset",
        type=str,
        default="ex_100.world.active.all.all.ppn_1.cls_2.tlf.ts_100.sl_15.gpt.aws_50.V0",
    )

    args = parser.parse_args()

    data_path = os.path.join(args.data_root, args.dataset)
    new_shard_size = args.new_shard_size

    new_ex = "ex_{}".format(new_shard_size)
    new_dataset = re.sub("ex_[0-9]*", new_ex, args.dataset)
    new_data_path = os.path.join(args.data_root, new_dataset)

    print("OLD data path: {}".format(data_path))
    print("NEW data path: {}\n".format(new_data_path))
    os.makedirs(new_data_path, exist_ok=True)

    merge(data_path, new_data_path, new_shard_size)
