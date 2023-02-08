import os
import sys
import random
from tqdm import tqdm
import torch
import numpy as np
import scipy

this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_path)
transformemnn_path = this_path.strip("data") + "/training/"
sys.path.append(transformemnn_path + "/db_dataloader")
droidlet_path = this_path.strip("data") + "/fairo/"
sys.path.append(droidlet_path)
sys.path.append(transformemnn_path + "/utils/")
sys.path.append(transformemnn_path)
from droidlet.lowlevel.minecraft.small_scenes_with_shapes import build_shape_scene
from droidlet.shared_data_structs import ErrorWithResponse
from query_builder import QAZoo
from memory_utils import build_memory, get_memory_text
from nsm_utils import make_world_args_from_config
from db_encoder import DBEncoder
from config_args import get_opts, read_configs
from active_agent_world import EpisodeRunner, WorldBuilder
from utils.generate_custom_val import generate_custom_val
from log_data import maybe_make_save_dir, update_data_info, save_files
from nsm_utils import NULL_MEMID
from interact import print_memory
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def print_num_choices(opts):
    num_choices = 0
    for i in range(opts.props_per_name):
        num_choices += scipy.special.comb(opts.num_props, i + 1)
    possible_samples = ((num_choices) ** opts.num_names) * (
        opts.num_names + opts.num_props
    )
    print("number of possible samples: {:.0f}\n".format(possible_samples))


def increment_clause_counts(clause_types, clause_type_counts):
    for clause_type in clause_types:
        clause_type_counts[clause_type] = clause_type_counts.get(clause_type, 0) + 1
        clause_type_counts["total_clauses"] += 1


def increment_conjunction_counts(conj_type, conj_type_counts):
    if conj_type:
        conj_type_counts[conj_type] += 1
    else:
        conj_type_counts["NONE"] += 1
    conj_type_counts["total_samples"] += 1


def increment_null_counts(is_null, null_counts):
    if is_null:
        null_counts["null_samples"] += 1
    null_counts["total_samples"] += 1


def check_update_proportions(
    args,
    configs,
    memids,
    query_type,
    clause_types,
    clause_type_counts,
    conj_type,
    conj_type_counts,
    null_counts,
    max_null_percent,
):
    """
    Checks query clause types to see if they are within the target distributions
    Returns reject_sample True/False
    """
    # Clause types
    if query_type == "FiltersQA":
        for clause_type in clause_types:
            target_percentage = configs[query_type]["clause_types"][clause_type]
            actual_percentage = (clause_type_counts.get(clause_type, 0) + 1) / (
                clause_type_counts["total_clauses"] + 1
            )
            if actual_percentage > target_percentage:
                return False

    # AND and NONE conjunctions
    if (args.num_clauses > 1) and (configs[query_type].get("operator_probs", False)):
        target_percentage = configs[query_type]["operator_probs"][conj_type]
        actual_percentage = (conj_type_counts.get(conj_type, 0) + 1) / (
            conj_type_counts["total_samples"] + 1
        )
        if actual_percentage > target_percentage:
            return False

    # NULL answers
    is_null = False
    if (not memids) or (memids[0] == NULL_MEMID):
        is_null = True
        actual_percentage = (null_counts["null_samples"] + 1) / (
            null_counts["total_samples"] + 1
        )
        if actual_percentage > max_null_percent:
            return False

    # TODO: sometimes, nearly all the refobjs are true. we should reject some of these
    # is_all = (len(memids[0]) >= len(db_dump['context']['reference_objects'])-2)
    # if is_all:

    # if it gets here, it passed all of the percentage checks and we will use this query
    increment_clause_counts(clause_types, clause_type_counts)
    increment_conjunction_counts(conj_type, conj_type_counts)
    increment_null_counts(is_null, null_counts)
    return True


def find_valid_query(
    args,
    Z,
    memory,
    snapshots,
    encoder,
    configs,
    clause_type_counts,
    conj_type_counts,
    null_counts,
    max_null_percent,
    verbose,
):
    """
    Given a single context memory, this searches for a query that doesn't doesn't go over the target proportions
    """
    search_counter = 0
    while search_counter < 10:  # search 10 different queries
        search_counter += 1
        try:
            db_dump = Z.get_qa(memory=memory, snapshots=snapshots)
        except (ErrorWithResponse, KeyError) as e:
            # print("ERROR: {}".format(e))
            # keep on truckin
            continue

        if not db_dump:
            if verbose:
                print("REJECTED: inactive clause")
            continue

        keep_sample = check_update_proportions(
            args,
            configs["QA"],
            db_dump["memids_and_vals"][0],
            db_dump["sample_query_type"],
            db_dump["sample_clause_types"],
            clause_type_counts,
            db_dump["sample_conjunction_type"],
            conj_type_counts,
            null_counts,
            max_null_percent,
        )

        if not keep_sample:
            if verbose:
                print("REJECTED: incorrect proportions")
            continue

        db_dump["context"]["text"] = get_memory_text(
            memory, db_dump
        )  # move to get_qa ?
        encoded_sample = encoder.encode_all(db_dump)

        return encoded_sample

    # couldn't find a valid query. Move on and generate a new scene
    return False


def main(opts):
    configs = opts.configs

    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    save_dir, metadata_file_name = maybe_make_save_dir(opts)

    if not opts.tokenizer_path:
        opts.tokenizer_path = save_dir

    if save_dir:
        print(save_dir)

    if not configs.get("WORLD"):
        print_num_choices(opts)
    else:
        world_opts = make_world_args_from_config(configs)
        opts.SL = world_opts.SL
        opts.H = world_opts.H
        print(
            'WARNING using config["WORLD"], ignoring opts.num_names, opts.num_props, opts.SL, etc.'
        )

    encoder = DBEncoder(
        query_encode=opts.query_encode,
        SL=opts.SL,
        COORD_SHIFT=(-opts.SL // 2, -opts.SL // 2, -opts.SL // 2),
        tokenizer=opts.tokenizer,
        tokenizer_path=opts.tokenizer_path,
        remove_instseg_locations=True,
    )
    examples = []
    info = {}

    num_iterations = opts.num_examples * 20

    pbar = None
    if opts.active_world_steps > 0:
        W = WorldBuilder()
    if not opts.disable_tqdm:
        pbar = tqdm(total=opts.num_examples, ncols=50)

    Z = QAZoo(opts, configs)

    clause_type_counts = {"total_clauses": 0}
    conj_type_counts = {"total_samples": 0, "AND": 0, "OR": 0, "NONE": 0}
    null_counts = {"total_samples": 0, "null_samples": 0}
    for i in range(num_iterations):
        world_spec = None
        if configs.get("WORLD"):
            world_opts = make_world_args_from_config(configs)
            world_spec = build_shape_scene(world_opts)

        if opts.active_world_steps > 0:
            a, w, lf, chat = W.instantiate_world_from_spec(configs, opts)
            runner = EpisodeRunner(a, snapshot_freq=50, keep_on_truckin=True)
            N = opts.active_world_steps
            if lf is not None:
                runner.run_episode(
                    min_steps=N, max_steps=N, logical_form=lf, chatstr=chat
                )
            else:
                runner.run_episode(min_steps=N, max_steps=N, no_agent_step=True)
            snapshots = runner.snapshots
            memory = a.memory
        else:
            snapshots = None
            memory = build_memory(opts, world_spec=world_spec)

        Z = QAZoo(opts, configs)

        if opts.make_custom_val:
            generate_custom_val(
                opts, configs, world_spec, snapshots, save_dir, memory, encoder
            )

        encoded_sample = find_valid_query(
            opts,
            Z,
            memory,
            snapshots,
            encoder,
            configs,
            clause_type_counts,
            conj_type_counts,
            null_counts,
            opts.max_percent_null,
            opts.verbose,
        )

        if encoded_sample:
            encoded_sample["world_configs"] = {"SL": opts.SL}
            if pbar:
                pbar.update(1)

            examples.append(encoded_sample)
            info = update_data_info(info, encoded_sample)
            if info["num_samples"] == opts.num_examples:
                break

    assert info["num_samples"] == opts.num_examples, "num_samples != opts.num_examples"

    print("{}\n{}\n{}".format(clause_type_counts, conj_type_counts, null_counts))

    if save_dir:
        save_files(examples, save_dir, encoder, opts, info, metadata_file_name)


if __name__ == "__main__":
    parser = get_opts()

    opts = parser.parse_args()
    opts.configs = read_configs(opts.config_file)

    main(opts)
