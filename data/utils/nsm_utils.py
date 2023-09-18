"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import droidlet.lowlevel.minecraft.shapes as shapes
from droidlet.lowlevel.minecraft.pyworld.item import GettableItem
from droidlet.interpreter.craftassist import DummyInterpreter
from droidlet.interpreter import AttributeInterpreter
from droidlet.lowlevel.minecraft.small_scenes_with_shapes import H
from copy import deepcopy
import time
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import (
    COLOR_BID_MAP,
)

# FIXME method to make sure nothing collides
NULL_MEMID = "a" * 32
AGENT_EID = 42
SPEAKER_EID = 11
UUID_HASH_VOCAB_SIZE = 2000
SPEAKER_NAME = "speaker"

# color info

EQA_COLOR_BID_MAP = deepcopy(COLOR_BID_MAP)

BLOCK_COLORS = [
    "aqua",
    "black",
    "blue",
    "fuchsia",
    "green",
    "gray",
    "lime",
    "maroon",
    "navy",
    "olive",
    "purple",
    "red",
    "silver",
    "teal",
    "white",
    "yellow",
    "orange",
    "brown",
    "pink",
    "gold",
]

ABSTRACT_SIZES = ["small", "medium", "large"]

# FIXME import from  small_scenes_with_shapes
SL = 15
HW = SL // 2
GROUND_DEPTH = SL // 2
GROUND_Y = 0
COORD_SHIFT = (-HW, 0, -HW)

"""
location values are in the range [1, SL-1]
the groudn level is at SL // 2
we reserve the 0th value for objects that have no location (e.g. null)

Negative: 1 2 3 4 5 6 7
Ground:   8
Positive: 9 10 11 12 13 14 15

"""

RADIUS = 4
MOB_NAMES = ["rabbit", "cow", "pig", "chicken", "sheep"]
# taken from mc wiki
EQA_MOB_COLORS = {
    "rabbit": ["white"],
    "cow": ["black", "white"],
    "pig": ["pink"],
    "chicken": ["white"],
    "sheep": ["gray", "black", "light gray"],
}

EQA_COLOR_MOB_MAP = {}

for mob in EQA_MOB_COLORS.keys():
    for color in EQA_MOB_COLORS[mob]:
        color_mob_map = EQA_COLOR_MOB_MAP.get(color, [])
        color_mob_map.append(mob)
        EQA_COLOR_MOB_MAP[color] = color_mob_map


SHAPES = {
    "cube": shapes.cube,  # size
    "sphere": shapes.sphere,  # radius
    "square": shapes.square,  # size
    "arch": shapes.arch,  # radius
    "triangle": shapes.triangle,  # radius
    "rectanguloid": shapes.rectanguloid,  # size
    "dome": shapes.dome,  # radius
    "circle": shapes.circle,  # radius
}
PLAYER_COLOR = ["dark brown", "light blue", "blue"]

unique_materials = set()
for color, materials in EQA_COLOR_BID_MAP.items():
    for m in materials:
        unique_materials.add(m)

MATERIALS_LIST = list(unique_materials)

# task info
POSSIBLE_TASKS = ["move", "build", "dance"]

# rel_direction info
REL_DIRECTIONS = ["LEFT", "RIGHT"]

# "singleton" items, not uniformly composable properties
GETTABLE_TEXTURES = ["fuzzy", "matte", "shiny"]
GETTABLE_TYPES = ["ball", "ring", "cone", "cross"]


def generate_items(N):
    items = []
    for i in range(N):
        tn = choice(GETTABLE_TYPES)
        props = [("has_tag", tn)]
        tx = choice(GETTABLE_TEXTURES)
        c = choice(BLOCK_COLORS)
        sz = choice(ABSTRACT_SIZES)
        for p in [tx, c, sz]:
            if choice([True, False]):
                props.append(("has_tag", p))
        items.append(GettableItem(tn, properties=props))
    return items


def choice(l):
    """returns a random entry from a list"""
    return l[torch.randint(len(l), (1,)).item()]


def k_multinomial(d):
    """returns a random key from a dict of the form {"key": float} with
    multinomial probs specified (up to a multiplicative constant
    by the float value for each key"""
    k, p = zip(*list(d.items()))
    return k[torch.multinomial(torch.Tensor(p), 1).item()]


# given a color, randomly select a bid with that color
def get_block_bid_with_color(color=None):
    if not color or color not in BLOCK_COLORS:
        color = choice(BLOCK_COLORS)
    return choice(EQA_COLOR_BID_MAP[color])


# given a block_id, meta find its color
def get_color_with_block_id(block_data=None):
    for key, val in EQA_COLOR_BID_MAP.items():
        if block_data in val:
            return key
    return None


# given a color, randomly select a mob with that color
def get_mob_name_with_color(color=None):
    if not color or color not in EQA_COLOR_MOB_MAP.keys():
        color = choice(EQA_COLOR_MOB_MAP.keys())
    return choice(EQA_COLOR_MOB_MAP[color])


def get_filter(agent_memory, filters_d):
    dummy_interpreter = DummyInterpreter(SPEAKER_NAME, None, agent_memory)
    return dummy_interpreter.subinterpret["filters"](
        dummy_interpreter, "SPEAKER", filters_d
    )


def get_attribute(agent_memory, attribute_d):
    dummy_interpreter = DummyInterpreter(SPEAKER_NAME, None, agent_memory)
    A = AttributeInterpreter()
    return A(dummy_interpreter, SPEAKER_NAME, attribute_d)


def append_time(times, old_time, text=""):
    times.append([time.time() - old_time, text])
    return time.time()


def format_location(x, y, z):
    return "({}, {}, {})".format(round(x), round(y), round(z))


def make_world_args_from_config(config):
    class Opt:
        pass

    opts = Opt
    c = config.get("WORLD", {})
    opts.SL = c.get("SL", SL)
    opts.H = c.get("H", H)
    opts.GROUND_DEPTH = c.get("GROUND_DEPTH", GROUND_DEPTH)
    opts.mob_config = c.get("mob_config", "")
    opts.MAX_NUM_SHAPES = c.get("MAX_NUM_SHAPES", 3)
    opts.MAX_NUM_GROUND_HOLES = c.get("MAX_NUM_GROUND_HOLES", 3)
    opts.fence = c.get("fence", False)
    opts.extra_simple = c.get("extra_simple", False)
    # TODO?
    opts.cuberite_x_offset = 0
    opts.cuberite_y_offset = 0
    opts.cuberite_z_offset = 0
    opts.iglu_scenes = c.get("iglu_scenes", "")
    opts.agent_lf_prob = config.get("AGENT", {}).get("lf_prob", 0.0)
    return opts


def check_physical(r):
    if r is not None and r.get("x"):
        if r["ref_type"] != "location" and r["ref_type"] != "attention":
            return True
    return False


def crossref_triples(snapshots):
    """
    find the triples associated to each reference_object in the snapshots
    we do this so we can do distances to inst_segs and other things with no names.
    """
    obj_texts = {}
    e = snapshots["num_steps"] - 1
    for memid, t in snapshots["triples"].items():
        if (
            t.get(e) is not None
            and t[e]["obj_text"]
            and not t[e]["obj_text"].startswith("_")
        ):
            subj = t[e].get("subj")
            if subj:
                r = snapshots["reference_objects"].get(subj, {})
                r = r.get(e)
                if check_physical(r):
                    if obj_texts.get(subj) is None:
                        obj_texts[subj] = []
                    obj_texts[subj].append(t[e]["obj_text"])

    return obj_texts


def maybe_name(snapshots, obj_texts, memid):
    """
    If the object has a name, return the name.
    Otherwise, it's a block, so return onoe of its properties.
    """
    last_time_slice = max(list(snapshots["reference_objects"][memid].keys()))
    name = snapshots["reference_objects"][memid][last_time_slice]["name"]
    if name and name != "none":
        return name, last_time_slice
    else:
        return choice(list(obj_texts[memid])), last_time_slice
