import numpy as np
from torch import randint
from nsm_utils import choice, k_multinomial, get_attribute, NULL_MEMID, crossref_triples, check_physical
from query_objects import QA
from droidlet.shared_data_struct import rotation 
from memory_utils import random_location

# FIXME: do all of these with FILTERS< fix the DSL to make it not annoying (e.g.
# w.r.t. "the nearest" not including that entity)

#############################################################################
"""
QUERY TYPES:
"closest_farthest_object": closest_farthest_object,
"closest_farthest_from_loc": closest_farthest_from_loc,
"max_direction": max_direction,
"distance_between": distance_between,
"""
#############################################################################

def maybe_name(snapshots, obj_texts, memid):
    """
    If the object has a name, return the name.
    Otherwise, it's a block, so return onoe of its properties.
    """
    last_time_slice = max(list(snapshots["reference_objects"][memid].keys()))
    name = snapshots["reference_objects"][memid][last_time_slice]["name"]
    if name and name != "none":
        return name
    else:
        return choice(list(obj_texts[memid]))

def distance_between(snapshots, query_configs, opts, obj_1=None, obj_2=None):
    """
    query of the form "how far is tag_1 from tag_2?"
    """
    # FIXME what if there are two blocks of the same type (e.g 2 holes). 
    # currently can't differentiate them
    obj_texts = crossref_triples(snapshots)
     # these can be the same- then its just 0 distance.  fixme?
    if not obj_1 and not obj_2:
        obj_1 = choice(list(obj_texts.keys()))
        obj_2 = choice(list(obj_texts.keys()))
    obj_1_desc = maybe_name(snapshots, obj_texts, obj_1)
    obj_2_desc = maybe_name(snapshots, obj_texts, obj_2)
    question_text = "how far is {} from {}?".format(obj_1_desc, obj_2_desc)
    memids = [obj_1, obj_2]
    r = snapshots["reference_objects"]
    last_time_slice = min( max(list(r[obj_1].keys())), max(list(r[obj_2].keys())) ) # last common time slice
    distance = ((r[obj_1][last_time_slice]["x"] - r[obj_2][last_time_slice]["x"])**2
                + (r[obj_1][last_time_slice]["y"] - r[obj_2][last_time_slice]["y"])**2
                + (r[obj_1][last_time_slice]["z"] - r[obj_2][last_time_slice]["z"])**2)
    answer_text = str(int(round(np.sqrt(distance))))
    return question_text, memids, answer_text


def closest_farthest_object(snapshots, query_configs, opts, obj_memid=None, polarity=None):
    """
    query of the form "what is the closest/farthest object from tag?"
    """
    obj_texts = crossref_triples(snapshots)
    if not obj_memid:
        obj_memid = choice(list(obj_texts.keys()))
        polarity = choice((1, -1))
    
    obj_desc = maybe_name(snapshots, obj_texts, obj_memid)
    pword = {-1:"closest", 1:"farthest"}[polarity]
    question_text = "which object is {} from {}?".format(pword, obj_desc)
    e = snapshots["num_steps"] - 1
    obj = snapshots["reference_objects"][obj_memid][e]
    dists = []
    for m, s in snapshots["reference_objects"].items():
        if m != obj_memid and s.get(e) is not None and check_physical(s[e]):
            dists.append((m,
                          (obj["x"]- s[e]["x"])**2 +
                          (obj["y"]- s[e]["y"])**2 +
                          (obj["z"]- s[e]["z"])**2
            ))
    dists = sorted(dists, key=lambda x: x[1], reverse=polarity==1)
    memid = dists[0][0]
    answer_text = maybe_name(snapshots, obj_texts, memid)
    return question_text, [memid], answer_text

def closest_farthest_from_loc(snapshots, query_configs, opts, pos=None, polarity=None):
    """
    query of the form "what is the closest/farthest object from (4, 5, 6)?"
    """
    obj_texts = crossref_triples(snapshots)
    ground_height = opts.SL//2
    if not pos:
        pos = random_location(opts, ground_height)
        polarity = choice((1, -1))
    pword = {-1:"closest", 1:"farthest"}[polarity]
    question_text = "which object is {} to ({})?".format(pword, " , ".join(map(str,pos)))
    e = snapshots["num_steps"] - 1
    dists = []
    for m, s in snapshots["reference_objects"].items():
        if s.get(e) is not None and check_physical(s[e]):
            dists.append((m,
                          (pos[0]- s[e]["x"])**2 +
                          (pos[1]- s[e]["y"])**2 +
                          (pos[2]- s[e]["z"])**2
            ))
    dists = sorted(dists, key=lambda x: x[1], reverse=polarity==1)
    memid = dists[0][0]
    answer_text = maybe_name(snapshots, obj_texts, memid)
    return question_text, [memid], answer_text


def max_direction(snapshots, query_configs, opts, dtype=None, coord=None, direction=None, frame=None):
    """
    Which object has the largest/smallest x/z?
    Which object is the farthest away from my right?
    """
    obj_texts = crossref_triples(snapshots)
    e = snapshots["num_steps"] - 1
    dists = []
    if not dtype:
        dtype = k_multinomial(
            {
            "cardinal": query_configs.get("cardinal", 1.0),
            "relative": query_configs.get("relative", 1.0)
            }
        )
    if dtype == "cardinal":
        #FIXME do "y" too?
        c = choice(["x", "z"])
        direction = choice([-1, 1])
        for m, s in snapshots["reference_objects"].items():
            if s.get(e) is not None and check_physical(s[e]):
                if s[e]["ref_type"] != "location":
                    dists.append((m, direction*(s[e][c])))
        dists = sorted(dists, key=lambda x: x[1], reverse=True)
        direction_word = {1:"largest", -1: "smallest"}[direction]
        question_text = "which object has the {} {}?".format(direction_word, c)
    elif dtype == "relative":
        direction = choice(["LEFT", "RIGHT", "FRONT", "BACK"])
        frame = choice(["your", "my"])
        nt = {"my": "Player", "your": "Self"}[frame]
        l = []
        for s in snapshots["reference_objects"].values():
            try:
                if s[e]["node_type"] == nt:
                    l.append((s[e]["x"], s[e]["y"], s[e]["z"], s[e]["yaw"], s[e]["pitch"]))
            except:
                e2 = list(s.keys())[-1]
                if s[e2]["node_type"] == nt:
                    l.append((s[e2]["x"], s[e2]["y"], s[e2]["z"], s[e2]["yaw"], s[e2]["pitch"]))
        l = l[0]
        x, y, z, yaw, pitch = l
        in_frame_dir = rotation.transform(rotation.DIRECTIONS[direction], yaw, pitch, inverted=True)
        dists = []
        b = {"x":x, "y":y, "z":z}
        for m, s in snapshots["reference_objects"].items():
            if s.get(e) is not None and check_physical(s[e]):
                if s[e]["ref_type"] != "location":
                    diff = np.array([s[e][l] - b[l] for l in ["x", "y", "z"]])
                    ip = diff@in_frame_dir
                    if ip > 0:
                        dists.append((m, ip))
        question_text = "which object is the farthest away from {} {}?".format(frame, direction.lower())
    if dists:
        dists = sorted(dists, key=lambda x: x[1], reverse=True)
        memid = dists[0][0]
        answer_text = maybe_name(snapshots, obj_texts, memid)
    else:
        memid = None
        answer_text = "none"
    return question_text, [memid], answer_text



class GeometricQA(QA):
    """
    closest/farthest
    most left/right/front/back
    distance between
    """
    def __init__(self, data, agent=None, memory=None, snapshots=None):
        super().__init__(data, agent=agent, memory=memory, snapshots=snapshots)
        assert(snapshots is not None)
        obj_texts = crossref_triples(snapshots)
        opts = data["opts"]
        configs = data["config"]
        qtype = k_multinomial(configs['question_types'])
        f = {
            "closest_farthest_object": closest_farthest_object,
            "closest_farthest_from_loc": closest_farthest_from_loc,
            "max_direction": max_direction,
            "distance_between": distance_between,
        }[qtype]
        query_configs = configs.get("question_configs",{}).get("qtype", {})
        #FIXME what if more than one?
        question_text, refobj_memids, answer_text = f(snapshots, query_configs, opts)
        self.question_text =  " {}".format(question_text)
        self.memids_and_vals = (refobj_memids, None)
        # FIXME
        self.question_logical_form = "NULL"
        # FIXME
        self.triple_memids_and_vals = ([NULL_MEMID], None)
        self.answer_text = "<|endoftext|>{}<|endoftext|>".format(answer_text)
        self.sample_clause_types = [qtype]
        self.sample_conjunction_type = None
