"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import torch
import random
import numpy as np


def generate_data_filename(opts):
    filename = ""
    filename += "ex_{}".format(opts.num_examples)
    filename += "." + opts.config_file.split("/")[-1].replace(".txt", "")

    if not opts.configs.get("WORLD"):
        filename += ".nms_{}.prps_{}".format(opts.num_names, opts.num_props)

    filename += ".ppn_{}.cls_{}".format(opts.props_per_name, opts.num_clauses)

    if opts.query_encode == "seq_lf":
        filename += ".slf"
    elif opts.query_encode == "tree_lf":
        filename += ".tlf"

    filename += ".ts_" + ("%.2f" % opts.triple_supervision_prob).replace(".", "")

    filename += ".sl_{}".format(opts.SL)

    filename += ".{}".format(opts.tokenizer)

    if opts.active_world_steps > 0:
        filename += ".aws_{}".format(opts.active_world_steps)

    if opts.data_name != "":
        filename += "." + opts.data_name

    print(filename)
    return filename


def maybe_make_save_dir(opts):
    save_dir = None
    metadata_file_name = None
    if opts.save_directory:
        if not os.path.exists(opts.save_directory):
            os.makedirs(opts.save_directory, exist_ok=True)

        filename = opts.save_filename
        if filename == "":
            # if filename not specified, create file name based on arg values
            filename = generate_data_filename(opts)
        save_dir = os.path.join(opts.save_directory, filename)
        data_file_name = os.path.join(save_dir, "train_val.pth")
        metadata_file_name = os.path.join(save_dir, "metadata.txt")
        if os.path.exists(data_file_name) and not opts.overwrite_data:
            print(
                "file exists. use --overwrite-data use_sample to overwrite. exiting\n"
            )
            exit()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    return save_dir, metadata_file_name


def update_data_info(info, encoded_sample):
    if not info:
        info = {}
        info["null_mem_num"] = 0
        info["sample_set"] = set()
        info["num_collisions"] = 0
        info["unique_samples"] = 0
        info["num_samples"] = 0
        info["triple_attns"] = []
        info["ref_obj_attns"] = []
        info["ref_obj_attns_without_null"] = []
        info["all_attns"] = []

    info["num_samples"] += 1
    encoded_query = encoded_sample["question"]["query_text"].data

    sample_triple_attns = encoded_sample["context"]["t_attn"].float().sum()
    sample_ref_obj_attns = encoded_sample["context"]["r_attn"].float().sum()
    sample_all_attns = sample_triple_attns + sample_ref_obj_attns
    info["triple_attns"].append(torch.log(sample_triple_attns))
    info["ref_obj_attns"].append(torch.log(sample_ref_obj_attns))
    info["all_attns"].append(torch.log(sample_all_attns))

    S = encoded_sample["context"]["r_attn"].size(0)  # num timesteps
    if encoded_sample["context"]["r_attn"][:, 0].sum().item() == S:
        info["null_mem_num"] += 1
    else:
        info["ref_obj_attns_without_null"].append(torch.log(sample_ref_obj_attns))
    return info


def print_and_summarize_data_info(opts, info):
    print("\n--------------------------------------\n")
    print(
        "duplicate samples: {}/{} ({:.1f}%)".format(
            info["num_collisions"],
            opts.num_examples,
            float(info["num_collisions"]) / opts.num_examples * 100,
        )
    )
    print(
        "unique samples: {}/{} ({:.1f}%)\n".format(
            info["unique_samples"],
            opts.num_examples,
            float(info["unique_samples"]) / opts.num_examples * 100,
        )
    )

    ref_obj_attns = np.array(info["ref_obj_attns"])
    ref_obj_attns_without_null = np.array(info["ref_obj_attns_without_null"])
    all_attns = np.array(info["all_attns"])

    messages_to_metadata = [
        "mean ln(ref_obj attns): {:.2f}".format(ref_obj_attns.mean()),
        "mean ln(ref_obj attns) without null: {:.2f}".format(
            ref_obj_attns_without_null.mean()
        ),
        "mean ln(all attns): {:.2f}".format(all_attns.mean()),
        "null mem samples: {}/{} ({:.1f}%)\n".format(
            info["null_mem_num"],
            len(info["ref_obj_attns"]),
            float(info["null_mem_num"]) / len(info["ref_obj_attns"]) * 100,
        ),
    ]
    for m in messages_to_metadata:
        print(m)
    return messages_to_metadata


def save_files(examples, save_dir, encoder, opts, info, metadata_file_name):
    random.shuffle(examples)

    # if not opts.no_val:
    #     data_file_name = os.path.join(save_dir, "train_val.pth")
    #     print("saving to: {}".format(data_file_name))
    #     torch.save(examples, data_file_name)

    metadata_messages = print_and_summarize_data_info(opts, info)
    metadata_file = open(metadata_file_name, "w")
    for m in metadata_messages:
        metadata_file.write(m + "\n")
    metadata_file.close()

    #### Splitting data into shards for faster memory loading ###
    if opts.no_val:
        num_train_samples = int(len(examples))
    else:
        num_train_samples = int(len(examples) * opts.split_ratio)

    train_examples = examples[:num_train_samples]
    val_examples = examples[num_train_samples:]

    shard_count = 1
    sample_idx = 0
    while sample_idx < len(train_examples):
        train_shard = train_examples[sample_idx : sample_idx + opts.shard_size]
        if opts.split == "test":
            data_file_name = os.path.join(save_dir, "test_" + str(opts.seed) + ".pth")
        elif opts.split == "val":
            data_file_name = os.path.join(save_dir, "val_" + str(opts.seed) + ".pth")
        else:
            data_file_name = os.path.join(
                save_dir, "train_" + str(opts.seed) + "_" + str(shard_count) + ".pth"
            )
        print("saving to: {}".format(data_file_name))
        torch.save(train_shard, data_file_name)
        shard_count += 1
        sample_idx += opts.shard_size

    if not opts.no_val:
        data_file_name = os.path.join(save_dir, "val.pth")
        print("saving to: {}".format(data_file_name))
        torch.save(val_examples, data_file_name)
