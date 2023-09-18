"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import json
import argparse


def read_configs(fname):
    with open(fname) as f:
        config = json.load(f)
    return config


# Shared argparser
def get_opts():
    parser = argparse.ArgumentParser()
    A = parser.add_argument
    A("--num_names", type=int, default=3)
    A("--num_props", type=int, default=6)
    A("--SL", type=int, default=15)
    A("--props_per_name", type=int, default=2)
    A(
        "--active_world_steps",
        type=int,
        default=0,
        help="if set > 0, runs an acitve world that many steps to generate data",
    )
    A("--overwrite-data", action="store_true", default=False)
    A(
        "--query_encode",
        choices=["seq_lf", "tree_lf"],
        default="tree_lf",
        help="input type (in addition to text)",
    )
    A("--num_examples", type=int, default=200)
    A("--num_clauses", type=int, default=1)
    A("--max_percent_null", type=int, default=0.1)
    A("--save_directory", type=str, default="", help="full path and file name")
    A(
        "--save_filename",
        type=str,
        default="",
        help="only path to save. file name will be created",
    )
    A("--data_name", type=str, default="", help="")
    A(
        "--flat_world",
        action="store_false",
        default=True,
        help="all entities at height 0",
    )
    A("--disable_tqdm", action="store_true", default=False)
    A("--triple_supervision", action="store_true", default=False)
    A(
        "--triple_supervision_prob",
        type=float,
        default=1.0,
        help="probability of using ts for each sample",
    )
    A("--max_output_tokens", type=float, default=1000)
    A("--verbose", action="store_true", default=False)
    A("--split_ratio", type=float, default=0.8, help="ratio to split train and val")
    A("--shard_size", type=int, default=100000)
    A(
        "--ignore_self",
        action="store_true",
        default=False,
        help="ignore self obj in memory searcher",
    )
    A(
        "--fix_clauses",
        action="store_true",
        default=False,
        help="use exactly num_clauses in each sample",
    )
    A("--not_prob", type=float, default=0.1)
    A("--and_prob", type=float, default=0.9)
    A("--single_clause_prob", type=float, default=0.5)
    A(
        "--tokenizer",
        choices=["bert", "simple", "gpt"],
        default="simple",
        help="text tokenizer",
    )
    A(
        "--tokenizer_path",
        type=str,
        default="",
        help="where to load the tokenizer words and idxs from",
    )
    A("--no_val", action="store_true", default=False, help="don't gen val set")
    A("--make_custom_val", action="store_true", default=False, help="")
    A("--seed", type=int, default=1)

    A("--config_file", type=str, default="data/config/default.txt", help="")

    A("--split", choices=["train", "val", "test"], default="train")

    return parser


def get_opts_and_config():
    opts = get_opts().parse_args()
    configs = read_configs(opts.config_file)
    return opts, configs
