import numpy as np
from query_objects import QA
from nsm_utils import (
    choice,
    k_multinomial,
    NULL_MEMID,
    crossref_triples,
    format_location,
    maybe_name,
)
from droidlet.shared_data_struct import rotation
import random
import math

LAST_TIME_SLICE = 1


#############################################################################
"""
QUERY TYPES:
"distance_moved": distance_moved,
"farthest_moved_object": farthest_moved_object,
"location_at_time": location_at_time,
"farthest_direction_moved_object": farthest_direction_moved_object,
"direction_moved_object": direction_moved_object
"""
#############################################################################


def location_at_time(snapshots, opts, o=None, time=None, name=None, memid=None):

    if not o:
        named_objects = [
            x
            for x in snapshots["reference_objects"].values()
            if (x.get(0, {}).get("name") is not None)
            and (x.get(LAST_TIME_SLICE, {}).get("name") is not None)
        ]
        o = choice(named_objects)
        time = choice(["beginning", "end"])
        name = o[0]["name"]
        memid = o[0]["uuid"]

    prep = "was"
    if time == "end":
        prep = "is"
    question_text = "where " + prep + " {} at the {} of the episode?".format(name, time)
    return question_text, memid, {"beginning": 0, "end": LAST_TIME_SLICE}[time], None


def distance_moved(snapshots, opts, memid=None):
    e = snapshots["num_steps"] - 1
    dists = []
    if memid is None:
        for m, s in snapshots["reference_objects"].items():
            if s.get(0) is not None and s.get(e) is not None:
                dists.append(
                    (
                        m,
                        math.sqrt(
                            (s[0]["x"] - s[e]["x"]) ** 2
                            + (s[0]["y"] - s[e]["y"]) ** 2
                            + (s[0]["z"] - s[e]["z"]) ** 2
                        ),
                    )
                )
        random.shuffle(dists)
        memid = dists[0][0]
        distance = dists[0][1]
        distance = int(distance)  # round
    else:
        s = snapshots["reference_objects"][memid]
        distance = math.sqrt(
            (s[0]["x"] - s[e]["x"]) ** 2
            + (s[0]["y"] - s[e]["y"]) ** 2
            + (s[0]["z"] - s[e]["z"]) ** 2
        )

    obj_texts = crossref_triples(snapshots)
    name, last_time_slice = maybe_name(snapshots, obj_texts, memid)

    question_text = "how far did {} move?".format(name)

    distance = str(int(distance))
    return question_text, memid, last_time_slice, distance


def farthest_moved_object(snapshots, opts):
    e = snapshots["num_steps"] - 1
    dists = []
    for m, s in snapshots["reference_objects"].items():
        if s.get(0) is not None and s.get(e) is not None:
            dists.append(
                (
                    m,
                    (s[0]["x"] - s[e]["x"]) ** 2
                    + (s[0]["y"] - s[e]["y"]) ** 2
                    + (s[0]["z"] - s[e]["z"]) ** 2,
                )
            )
    dists = sorted(dists, key=lambda x: x[1], reverse=True)
    question_text = "which object moved the farthest?"
    memid = dists[0][0]
    return question_text, memid, LAST_TIME_SLICE, None


def direction_moved_object(
    snapshots, opts, dtype=None, c=None, direction=None, frame=None
):
    """
    Which object increased x
    Which object moved to my left? (which object moved in the direction of my left)
    """
    e = snapshots["num_steps"] - 1
    memids = []
    if not dtype:
        dtype = k_multinomial(
            {
                "cardinal": opts.get("cardinal", 1.0),
                "relative": opts.get("relative", 1.0),
            }
        )
    if dtype == "cardinal":
        if not c:
            c = choice(["x", "z"])
            direction = choice([-1, 1])
        for m, s in snapshots["reference_objects"].items():
            if s.get(0) is not None and s.get(e) is not None:
                if (direction * (s[e][c] - s[0][c])) > 0:
                    memids.append(m)
        direction_word = {1: "increased", -1: "decreased"}[direction]
        question_text = "which object {} {}?".format(direction_word, c)
    elif dtype == "relative":
        if not direction:
            direction = choice(["LEFT", "RIGHT", "FRONT", "BACK"])
            frame = choice(["your", "my"])
        nt = {"my": "Player", "your": "Self"}[frame]
        yaw_pitch = []
        for s in snapshots["reference_objects"].values():
            key = list(s.keys())[0]  # some don't have key (timestep) 0
            if s[key]["node_type"] == nt:
                yaw_pitch.append((s[key]["yaw"], s[key]["pitch"]))
        yaw_pitch = yaw_pitch[0]
        in_frame_dir = rotation.transform(
            rotation.DIRECTIONS[direction], *yaw_pitch, inverted=True
        )
        for m, s in snapshots["reference_objects"].items():
            if s.get(0) is not None and s.get(e) is not None:
                diff = np.array([s[e][l] - s[0][l] for l in ["x", "y", "z"]])
                ip = diff @ in_frame_dir  # dot the difference with the
                if ip > 0:
                    memids.append(m)
        question_text = "which object moved to {} {}?".format(frame, direction.lower())

    return question_text, memids, LAST_TIME_SLICE, None


def farthest_direction_moved_object(
    snapshots, opts, dtype=None, c=None, direction=None, frame=None
):
    """
    Which object increased x the most
    Which object moved to my left the most? (which object moved in the direction of my left the most)
    """
    e = snapshots["num_steps"] - 1
    dists = []
    if not dtype:
        dtype = k_multinomial(
            {
                "cardinal": opts.get("cardinal", 1.0),
                "relative": opts.get("relative", 1.0),
            }
        )
    if dtype == "cardinal":
        if not c:
            c = choice(["x", "z"])
            direction = choice([-1, 1])
        for m, s in snapshots["reference_objects"].items():
            if s.get(0) is not None and s.get(e) is not None:
                dists.append((m, direction * (s[e][c] - s[0][c])))
        dists = sorted(dists, key=lambda x: x[1], reverse=True)
        direction_word = {1: "increased", -1: "decreased"}[direction]
        question_text = "which object {} {} the most?".format(direction_word, c)
    elif dtype == "relative":
        if not direction:
            direction = choice(["LEFT", "RIGHT", "FRONT", "BACK"])
            frame = choice(["your", "my"])
        nt = {"my": "Player", "your": "Self"}[frame]
        yaw_pitch = []
        for s in snapshots["reference_objects"].values():
            key = list(s.keys())[0]  # some don't have key (timestep) 0
            if s[key]["node_type"] == nt:
                yaw_pitch.append((s[key]["yaw"], s[key]["pitch"]))
        yaw_pitch = yaw_pitch[0]
        in_frame_dir = rotation.transform(
            rotation.DIRECTIONS[direction], *yaw_pitch, inverted=True
        )
        dists = []
        for m, s in snapshots["reference_objects"].items():
            if s.get(0) is not None and s.get(e) is not None:
                diff = np.array([s[e][l] - s[0][l] for l in ["x", "y", "z"]])
                ip = diff @ in_frame_dir  # dot the difference with the
                if ip > 0:
                    dists.append((m, ip))
        question_text = "which object moved to {} {} the most?".format(
            frame, direction.lower()
        )
    if dists:
        dists = sorted(dists, key=lambda x: x[1], reverse=True)
        memid = dists[0][0]
    else:
        memid = None
    return question_text, memid, LAST_TIME_SLICE, None


def get_output_from_mem(
    refobj_memids, snapshots, output_type, random_sample=False, time_slice=1
):
    """
    returns the answer text

    refobj_memids (list)
    """
    if output_type == "MEMORY":
        return "<|endoftext|> MEMORY"
    elif output_type == "location":
        locs = []
        for refobj_memid in refobj_memids:
            r = snapshots["reference_objects"][refobj_memid][time_slice]
            locs.append(format_location(r["x"], r["y"], r["z"]))
        locs = " ".join(locs) if locs else "None"
        return locs
    elif output_type == "has_tag":
        st = snapshots["triples"]

        tags = []
        for refobj_memid in refobj_memids:
            tags.append(
                [
                    v[time_slice]["obj_text"]
                    for v in st.values()
                    if v[time_slice]["subj"] == refobj_memid
                ]
            )

        if len(tags) == 0:
            return "None"

        tags = list(set([t for t in tags if t[0] != "_"]))
        if random_sample:
            return np.random.choice(tags)
        else:
            return " ".join(tags)
    elif output_type == "name":
        # TODO backoff to has_name triple
        names = []
        for refobj_memid in refobj_memids:
            name = snapshots["reference_objects"][refobj_memid][time_slice]["name"]
            if name:
                names.append(name)

        names = " ".join(names) if names else "None"
        return names
    else:
        raise Exception("bad output type in temporal query {}".format(output_type))


# TODO: use these in clauses
class TemporalSnapshotQA(QA):
    """
    QA object using premade snapshots
    """

    def __init__(self, data, agent=None, memory=None, snapshots=None):
        super().__init__(data, agent=agent, memory=memory, snapshots=snapshots)
        assert snapshots is not None
        opts = data["opts"]
        configs = data["config"]
        qtype = k_multinomial(configs["question_types"])
        f = {
            "distance_moved": distance_moved,
            "farthest_moved_object": farthest_moved_object,
            "location_at_time": location_at_time,
            "farthest_direction_moved_object": farthest_direction_moved_object,
            "direction_moved_object": direction_moved_object,
        }[qtype]
        self.sample_clause_types = [qtype]
        self.sample_conjunction_type = None
        query_configs = configs.get("question_configs", {}).get("qtype", {})
        # WARNING only handles 1 at a time
        question_text, refobj_memids, time_slice, answer = f(snapshots, query_configs)
        self.question_text = " {}".format(question_text)
        # WARNING should continually check to see if it's a list
        if type(refobj_memids) is not list:
            refobj_memids = [refobj_memids]

        self.memids_and_vals = (refobj_memids, None)

        self.question_logical_form = "NULL"
        self.triple_memids_and_vals = ([NULL_MEMID], None)
        if qtype == "location_at_time":
            output_type = "location"
        else:
            output_type = k_multinomial(configs["output_prop_probs"])
        if answer is None:
            answer = get_output_from_mem(
                refobj_memids, snapshots, output_type, time_slice=time_slice
            )

        self.answer_text = "<|endoftext|>{}<|endoftext|>".format(answer)
