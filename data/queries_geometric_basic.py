from utils.nsm_utils import (
    choice,
    k_multinomial,
    NULL_MEMID,
    crossref_triples,
)
from query_objects import QA
import random
import math


def where_is_object(snapshots, opts, query_configs, frame=None):
    """
    Where am I?
    Where are you?

    Returns: location
    """
    if frame is None:
        frame = choice(["you", "me"])
    nt = {"me": "Player", "you": "Self"}[frame]
    verb = {"me": "am", "you": "are"}[frame]
    frame_ref = {"me": "i", "you": "you"}[frame]
    e = snapshots["num_steps"] - 1

    for m, s in snapshots["reference_objects"].items():
        try:
            if s[e]["node_type"] == nt:
                loc_x, loc_y, loc_z, pitch, yaw = (
                    s[e]["x"],
                    s[e]["y"],
                    s[e]["z"],
                    s[e]["pitch"],
                    s[e]["yaw"],
                )
                memid = m
        except:
            e2 = list(s.keys())[-1]
            if s[e2]["node_type"] == nt:
                loc_x, loc_y, loc_z, pitch, yaw = (
                    s[e2]["x"],
                    s[e2]["y"],
                    s[e2]["z"],
                    s[e2]["pitch"],
                    s[e2]["yaw"],
                )
                memid = m
    question_text = "where {} {}?".format(verb, frame_ref)

    loc_x, loc_y, loc_z = (
        format_loc(loc_x, opts.SL),
        format_loc(loc_y, opts.SL),
        format_loc(loc_z, opts.SL),
    )
    answer_text = "({}, {}, {})".format(loc_x, loc_y, loc_z)
    return question_text, memid, answer_text


def look_direction(snapshots, opts, query_configs, frame=None):
    """
    Where are you looking?
    Where am I looking?

    Returns: pitch/yaw
    """
    if frame is None:
        frame = choice(["you", "me"])
    nt = {"me": "Player", "you": "Self"}[frame]
    verb = {"me": "am", "you": "are"}[frame]
    frame_ref = {"me": "i", "you": "you"}[frame]
    e = snapshots["num_steps"] - 1

    for m, s in snapshots["reference_objects"].items():
        try:
            if s[e]["node_type"] == nt:
                x, y, z, pitch, yaw = (
                    s[e]["x"],
                    s[e]["y"],
                    s[e]["z"],
                    s[e]["pitch"],
                    s[e]["yaw"],
                )
                memid = m
        except:
            e2 = list(s.keys())[-1]
            if s[e2]["node_type"] == nt:
                x, y, z, pitch, yaw = (
                    s[e2]["x"],
                    s[e2]["y"],
                    s[e2]["z"],
                    s[e2]["pitch"],
                    s[e2]["yaw"],
                )
                memid = m

    question_text = "where {} {} looking?".format(verb, frame_ref)
    answer_text = "({:.2f}, {:.2f})".format(pitch, yaw)
    return question_text, memid, answer_text


def round_to_base(x, base=1):
    return base * round(x / base)


def format_loc(loc, SL):
    loc = round(loc)
    loc = max(loc, 0)
    loc = min(loc, SL)
    return loc


def what_is_location_nearby(
    snapshots, opts, query_configs, direction=None, frame=None, steps=None
):
    """
    What is the location 2 steps to my left?
    What is the location 4 steps to your front?


    Returns: location
    """
    if direction is None:
        direction = choice(["LEFT", "RIGHT", "FRONT", "BACK"])
        frame = choice(["me", "you"])
        steps = random.randint(1, 5)  # only physical steps between 1 and 5

    nt = {"me": "Player", "you": "Self"}[frame]
    verb = {"me": "my", "you": "your"}[frame]
    e = snapshots["num_steps"] - 1
    memid = None

    for m, s in snapshots["reference_objects"].items():
        try:
            if s[e]["node_type"] == nt:
                x, y, z, pitch, yaw = (
                    s[e]["x"],
                    s[e]["y"],
                    s[e]["z"],
                    s[e]["pitch"],
                    s[e]["yaw"],
                )
                memid = m
        except:
            e2 = list(s.keys())[-1]
            if s[e2]["node_type"] == nt:
                x, y, z, pitch, yaw = (
                    s[e2]["x"],
                    s[e2]["y"],
                    s[e2]["z"],
                    s[e2]["pitch"],
                    s[e2]["yaw"],
                )
                memid = m
            else:
                return False, False, False

    question_text = "what is the location {} steps to {} {}?".format(
        steps, verb, direction.lower()
    )

    loc_x, loc_y, loc_z = x, y, z  # new vals to be overwritten

    # METHOD 1: find from radians.
    # Add direction, and then snap to grid (round to int)

    # convert to [0, 2*pi] range (expected to be in [-pi, pi] range)
    yaw_rads = yaw
    if yaw_rads < 0:
        yaw_rads = 2 * math.pi - abs(yaw_rads)

    # Get direction in radians
    if direction == "LEFT":
        yaw_rads = (yaw_rads + (math.pi / 2)) % (math.pi * 2)
    elif direction == "RIGHT":
        yaw_rads = (yaw_rads + (3 * math.pi / 2)) % (math.pi * 2)
    elif direction == "BACK":
        yaw_rads = (yaw_rads + math.pi) % (math.pi * 2)

    # convert to cartesian
    x_add = math.cos(yaw_rads)
    z_add = math.sin(yaw_rads)

    # Step in the direction
    loc_x += steps * x_add
    loc_z += steps * z_add

    loc_x, loc_y, loc_z = (
        format_loc(loc_x, opts.SL),
        format_loc(loc_y, opts.SL),
        format_loc(loc_z, opts.SL),
    )

    # METHOD 2: SNAP TO GRID THEN ADD
    # snap to {0, pi/2, pi, -pi, -pi/2}
    # yaw_snapped = round_to_base(yaw, math.pi/2)
    # if yaw_snapped == -math.pi:
    #     yaw_snapped = math.pi

    # loc_x, loc_y, loc_z = x, y, z # new vals to be overwritten
    # if direction == 'LEFT':
    #     if yaw_snapped == 0:
    #         loc_z += steps
    #     elif yaw_snapped == math.pi/2:
    #         loc_x -= steps
    #     elif yaw_snapped == math.pi:
    #         loc_z -= steps
    #     elif yaw_snapped == -math.pi/2:
    #         loc_x += steps
    #     else:
    #         raise NameError('Incorrect snapping')
    # elif direction == 'RIGHT':
    #     if yaw_snapped == 0:
    #         loc_z -= steps
    #     elif yaw_snapped == math.pi/2:
    #         loc_x += steps
    #     elif yaw_snapped == math.pi:
    #         loc_z += steps
    #     elif yaw_snapped == -math.pi/2:
    #         loc_x -= steps
    #     else:
    #         raise NameError('Incorrect snapping')
    # elif direction == 'FRONT':
    #     if yaw_snapped == 0:
    #         loc_x += steps
    #     elif yaw_snapped == math.pi/2:
    #         loc_z += steps
    #     elif yaw_snapped == math.pi:
    #         loc_x -= steps
    #     elif yaw_snapped == -math.pi/2:
    #         loc_z -= steps
    #     else:
    #         raise NameError('Incorrect snapping')
    # elif direction == 'BACK':
    #     if yaw_snapped == 0:
    #         loc_x -= steps
    #     elif yaw_snapped == math.pi/2:
    #         loc_z -= steps
    #     elif yaw_snapped == math.pi:
    #         loc_x += steps
    #     elif yaw_snapped == -math.pi/2:
    #         loc_z += steps
    #     else:
    #         raise NameError('Incorrect snapping')

    answer_text = "({}, {}, {})".format(int(loc_x), int(loc_y), int(loc_z))

    return question_text, memid, answer_text


class BasicGeometricQA(QA):
    """
    where are you?
    where am i looking?
    what is the location 2 steps to my left?
    """

    def __init__(self, data, agent=None, memory=None, snapshots=None):
        super().__init__(data, agent=agent, memory=memory, snapshots=snapshots)
        assert (
            snapshots is not None
        )  # asserting multiple snapshots. not necessary for these queries.
        obj_texts = crossref_triples(snapshots)
        opts = data["opts"]
        configs = data["config"]
        qtype = k_multinomial(configs["question_types"])
        f = {
            "where_is_object": where_is_object,
            "look_direction": look_direction,
            "what_is_location_nearby": what_is_location_nearby,
        }[qtype]
        query_configs = configs.get("question_configs", {}).get("qtype", {})
        question_text, refobj_memids, answer_text = f(snapshots, opts, query_configs)
        if not question_text:
            self.question_text = False
        else:
            self.question_text = " {}".format(question_text)
            self.memids_and_vals = (refobj_memids, None)
            self.question_logical_form = "NULL"
            self.triple_memids_and_vals = ([NULL_MEMID], None)
            self.answer_text = "<|endoftext|>{}<|endoftext|>".format(answer_text)
            self.sample_clause_types = [qtype]
            self.sample_conjunction_type = None
