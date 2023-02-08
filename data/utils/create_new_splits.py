import argparse
import os
import torch
import re


def reformat_number(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "%d%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


def create_split(data_path, new_data_path, new_data_size):
    """
    This file creates new data splits from an original dataset
    """
    total_len = 0
    for filename in os.listdir(data_path):
        if "train_" in filename:
            f = os.path.join(data_path, filename)
            if os.path.isfile(f):
                shard = torch.load(f)
                if new_data_size == 1000:
                    shard = shard[0:1000]
                total_len += len(shard)
                if total_len > new_data_size:
                    return
                new_shard_file = os.path.join(new_data_path, filename)
                print("saving shard {}".format(new_shard_file))
                torch.save(shard, new_shard_file)

        elif "val" in filename or "test" in filename:
            f = os.path.join(data_path, filename)
            if os.path.isfile(f):
                shard = torch.load(f)
                new_shard_file = os.path.join(new_data_path, filename)
                print("saving shard {}".format(new_shard_file))
                torch.save(shard, new_shard_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    A = parser.add_argument
    A("--new_data_size", type=int, default=100000)
    A(
        "--data_root",
        type=str,
        default="data/data_new/",
        help="",
    )
    A(
        "--dataset",
        type=str,
        default="ex_10000.world.filters_only.ppn_1.cls_2.tlf.ts_100.sl_15.gpt.1M.FINAL",
        help="",
    )

    args = parser.parse_args()

    data_path = os.path.join(args.data_root, args.dataset)
    new_data_size = args.new_data_size

    new_data_size_name = reformat_number(new_data_size)
    new_dataset = re.sub("1M[0-9]*", new_data_size_name, args.dataset)
    new_data_path = os.path.join(args.data_root, new_dataset)

    print("OLD data path: {}".format(data_path))
    print("NEW data path: {}\n".format(new_data_path))
    os.makedirs(new_data_path, exist_ok=True)

    create_split(data_path, new_data_path, new_data_size)
