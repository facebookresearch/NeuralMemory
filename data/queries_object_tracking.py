"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from query_objects import QA
from utils.nsm_utils import (
    choice,
    k_multinomial,
    NULL_MEMID,
    format_location,
)
from utils.memory_utils import random_location

LAST_TIME_SLICE = 1


#############################################################################
"""
if I/you move to <loc>, where will the <item> be?
NOT DONE how many <item> do I/you have?
NOT DONE If I put <n> item <loc> how many will there be at <loc>?
NOT DONE who has the <item>

"""
#############################################################################


def hypothetical_location(snapshots, opts):
    e = snapshots["num_steps"] - 1
    carrier = choice(["you", "I"])
    items = [
        t[0]
        for t in snapshots["reference_objects"].values()
        if t.get(1) and t[1]["ref_type"] == "item_stack"
    ]
    item = choice(items)
    tags = [
        t[0]["obj_text"]
        for t in snapshots["triples"].values()
        if t.get(1) and t[1]["subj"] == item["uuid"]
    ]
    non_underscore_tags = [t for t in tags if t[0] != "_"]
    loc = random_location(opts)
    formatted_tags = non_underscore_tags[0]
    for t in non_underscore_tags[1:]:
        formatted_tags = t + ", " + formatted_tags
    question_text = "if {} move to {}, where will the {} be?".format(
        carrier, format_location(*loc), formatted_tags
    )
    owner = ""
    # WARNING only handles 1 agent and 1 player
    if "_in_others_inventory" in tags:
        owner = "I"
    elif "_in_inventory" in tags:
        owner = "you"
    if not owner or owner != carrier:
        answer = format_location(item["x"], item["y"], item["z"])
    else:
        answer = format_location(*loc)
    return question_text, None, answer


class ObjectTrackingQA(QA):
    """
    QA object using premade snapshots, asking questions about the location
    of pickable objects

    if I/you move to <loc>, where will the <item> be?
    NOT DONE how many <item> do I/you have?
    NOT DONE If I put <n> item <loc> how many will there be at <loc>?
    NOT DONE who has the <item>
    """

    def __init__(self, data, agent=None, memory=None, snapshots=None):
        super().__init__(data, agent=agent, memory=memory, snapshots=snapshots)
        assert snapshots is not None
        opts = data["opts"]
        configs = data["config"]
        qtype = "hypothetical_location"
        f = {
            "hypothetical_location": hypothetical_location,
        }[qtype]
        self.sample_clause_types = [qtype]
        self.sample_conjunction_type = None
        query_configs = configs.get("question_configs", {}).get("qtype", {})
        query_configs["SL"] = opts.SL
        # WARNING can only handle one at a time
        question_text, refobj_memids, answer = f(snapshots, query_configs)
        self.question_text = " {}".format(question_text)

        # WARNING only checks to see if it's a list once
        if type(refobj_memids) is not list:
            refobj_memids = [refobj_memids]

        self.memids_and_vals = (refobj_memids, None)

        self.question_logical_form = "NULL"
        self.triple_memids_and_vals = ([NULL_MEMID], None)
        self.answer_text = "<|endoftext|>{}<|endoftext|>".format(answer)
