import numpy as np
from droidlet.interpreter.tests.all_test_commands import command
from nsm_utils import k_multinomial, choice

# FIXME PUT these in all_test_commands so they will be updated as
# grammar changes


COMMAND_PROBS = {
    "move_around": 1.0,
    "build": 1.0,
    "dig": 1.0,
    "follow": 1.0,
    "move_loc": 1.0,
    "destroy": 1.0,
    "give_get_bring": 1.0,
}


def get_command(obj_data, command_probs=COMMAND_PROBS):
    BUILDERS = {
        "move_around": make_move_around,
        "build": make_build,
        "dig": make_dig,
        "follow": make_follow,
        "give_get_bring": make_get_give_bring,
        "move_loc": make_move_loc,
        "destroy": make_destroy,
    }
    return BUILDERS[k_multinomial(command_probs)](obj_data)


# TODO more opts
def get_rel_obj_location(obj_data):
    rel = k_multinomial(
        {
            "none": 4.0,
            "LEFT": 1.0,
            "RIGHT": 1.0,
            "FRONT": 1.0,
            "BACK": 1.0,
        }
    )

    t = choice(["obj", "agent", "player", "there"])
    if t == "obj":
        obj = choice(obj_data["all_objs"])
        F = {
            "filters": {
                "where_clause": {"AND": [{"pred_text": "has_tag", "obj_text": obj}]}
            }
        }
        L = {"location": {"reference_object": F}}
        obj = "the " + obj
    elif t == "agent":
        obj = "you"
        L = {"location": {"reference_object": {"special_reference": "AGENT"}}}
    elif t == "player":
        obj = "me"
        L = {"location": {"reference_object": {"special_reference": "SPEAKER"}}}
    else:
        obj = "there"
        L = {"location": {"reference_object": {"special_reference": "SPEAKER_LOOK"}}}
    if rel != "none":
        L["location"]["relative_direction"] = rel
        text = "to the " + rel.lower() + " of " + obj
    else:
        text = obj
    return L, text


def make_follow(obj_data):
    name = choice(obj_data["mobs"])
    LF = {
        "event_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {
                        "filters": {
                            "where_clause": {
                                "AND": [{"pred_text": "has_name", "obj_text": name}]
                            }
                        }
                    }
                },
            }
        ],
        "terminate_condition": {"condition": "NEVER"},
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    }
    return LF, "follow {}".format(name)


# TODO N times
def make_move_around(obj_data):
    tag = choice(obj_data["all_objs"])
    dance_loc = {
        "relative_direction": "AROUND",
        "reference_object": {
            "filters": {
                "where_clause": {"AND": [{"pred_text": "has_tag", "obj_text": tag}]}
            }
        },
    }
    LF = {"action_type": "DANCE", "location": dance_loc}
    return command(LF), "go around the {}".format(tag)


# TODO get, drop
def make_get_give_bring(obj_data):
    if not obj_data.get("items"):
        return None, ""
    item = choice(obj_data["items"])
    a = [{"pred_text": "has_tag", "obj_text": item["typeName"]}]
    item_description_text = item["typeName"]
    if np.random.rand() > 0.5 and item["properties"]:
        p = choice(item["properties"])
        p = p[1]
        if p != item["typeName"]:
            item_description_text = p + " " + item_description_text
            a.append({"pred_text": "has_tag", "obj_text": p})
    obj_to_get = {"filters": {"where_clause": {"AND": a}}}
    r_lf, r_text = get_rel_obj_location(obj_data)
    G_LF = {"action_type": "GET", "reference_object": obj_to_get, "receiver": r_lf}

    text = "bring the {} to {}".format(item_description_text, r_text)
    return command(G_LF), text


def make_move_loc(obj_data):
    r_lf, r_text = get_rel_obj_location(obj_data)
    move_LF = {"action_type": "MOVE", "location": r_lf["location"]}
    text = "move to " + r_text
    return command(move_LF), text


def make_destroy(obj_data):
    if not obj_data.get("inst_segs"):
        return None, ""
    tag = choice(obj_data["inst_segs"])
    R = {
        "filters": {
            "where_clause": {
                "AND": [
                    {"pred_text": "has_tag", "obj_text": tag},
                ]
            }
        },
    }
    D_LF = {
        "action_type": "DESTROY",
        "reference_object": R,
    }
    text = "destroy the {}".format(tag)
    return command(D_LF), text


def make_build(obj_data):
    # TODO optional color
    sname = choice(["cube", "sphere"])
    S = {
        "filters": {
            "where_clause": {
                "AND": [
                    {"pred_text": "has_name", "obj_text": sname},
                    {"pred_text": "has_size", "obj_text": "small"},
                ]
            }
        },
    }
    r_lf, r_text = get_rel_obj_location(obj_data)
    B_LF = {"action_type": "BUILD", "schematic": S, "location": r_lf["location"]}
    text = "build a {} at {}".format(sname, r_text)
    return command(B_LF), text


def make_dig(obj_data):
    r_lf, r_text = get_rel_obj_location(obj_data)
    dig_LF = {
        "schematic": {
            "filters": {
                "where_clause": {"AND": [{"pred_text": "has_name", "obj_text": "hole"}]}
            }
        },
        "action_type": "DIG",
    }
    dig_LF["location"] = r_lf["location"]
    text = "dig a hole at " + r_text
    return command(dig_LF), text
