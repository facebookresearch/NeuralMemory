import torch
from copy import deepcopy
from nsm_utils import choice, k_multinomial, get_attribute, NULL_MEMID
from memory_utils import random_location
from query_objects import QA


# TODO LEFT/RIGHT/FRAMES
DMAX_FRAC = 2
DMIN_FRAC = 8


#############################################################################
# each of the clause generators returns
# a clause "executable", to be run by basic_search
# a clause lf, to be fed to the logical_form encoder and then to the model
# a clause text encoding, to be fed to the text encoder and then to the model

#############################################################################
"""
QUERY TYPES:

"name": build_name_clause,
"tag": build_tag_clause,
"absolute_cardinal": build_absolute_cardinal_clause,
"absolute_distance": build_absolute_distance_clause,
"direction": build_direction_clause,

"""
#############################################################################


def get_universe_from_memory(memory):
    if getattr(memory, "names_and_properties", None):
        return memory.names_and_properties
    else:
        #! WARNING if the universe is not specified in the memory,
        # universe will just be all that is in memory, not all possibilities
        names = memory._db_read("SELECT DISTINCT name FROM ReferenceObjects")
        props = memory._db_read(
            "SELECT DISTINCT obj_text FROM Triples WHERE pred_text=?", "has_tag"
        )
        return {
            "names": [x[0] for x in names if x[0] is not None],
            "props": [x[0] for x in props if x[0] is not None],
        }


def build_absolute_cardinal_clause(
    data, agent_memory, opts, cardinal=None, ctype=None, value=None
):
    """
    where the x coordinate is greater than 5
    """

    if cardinal is None:
        dimensions = ["x", "z"] if opts.flat_world else ["x", "y", "z"]
        cardinal = choice(dimensions)
    if ctype is None:
        ctype = choice(["GREATER_THAN", "LESS_THAN"])

    if value is None:
        if ctype == "GREATER_THAN":
            lower = 0
            upper = opts.SL - 1
        else:
            lower = 1
            upper = opts.SL
        value = str(torch.randint(lower, upper, (1,)).item())

    clause = {
        "input_left": {"attribute": cardinal},
        "comparison_type": ctype,
        "input_right": value,
    }
    clause_text = "where the {} coordinate is {} than {}".format(
        cardinal, ctype[:-5].lower(), value
    )

    return clause, deepcopy(clause), clause_text, None


def build_absolute_distance_clause(
    data, agent_memory, opts, ctype=None, pos=None, value=None
):
    """
    where the distance to distance to (0,0,0) is greater than 5
    """
    ground_height = 5  # FIXME should be opts.SL//2
    if ctype is None:
        ctype = choice(["GREATER_THAN", "LESS_THAN"])
    if pos is None:
        pos = random_location(opts, ground_height)
    if value is None:
        value = str(
            torch.randint(opts.SL // DMIN_FRAC, opts.SL // DMAX_FRAC, (1,)).item()
        )
    clause_text = "where the distance to ({}) is {} than {}".format(
        " , ".join(map(str, pos)), ctype[:-5].lower(), value
    )
    distance_lf = {
        "linear_extent": {
            "relative_direction": "AWAY",
            "source": {
                "reference_object": {
                    "special_reference": {"coordinates_span": str(pos)}
                }
            },
        }
    }
    distance_attribute = get_attribute(agent_memory, distance_lf)
    clause_executable = {
        "input_left": {"attribute": distance_attribute},
        "comparison_type": ctype,
        "input_right": value,
    }
    clause_lf = {
        "input_left": {"attribute": distance_lf},
        "comparison_type": ctype,
        "input_right": value,
    }

    return clause_executable, clause_lf, clause_text, None


def build_direction_clause(data, agent_memory, opts, direction=None, frame=None):

    if not direction:
        copts = opts.clause_opts_from_config.get("direction", {})
        dtype = copts.get("direction_types", "all")
        if dtype == "all":
            direction = choice(["LEFT", "RIGHT", "UP", "DOWN", "FRONT", "BACK"])
        elif dtype == "horizontal":
            direction = choice(["LEFT", "RIGHT", "FRONT", "BACK"])
        elif dtype == "vertical":
            direction = choice(["ABOVE", "BELOW"])
        else:
            raise Exception(
                "unknown direction type in build_direction_clause {}".format(dtype)
            )
    if not frame:
        frame = choice(["AGENT", "SPEAKER"])
    direction_lf = {
        "linear_extent": {
            "relative_direction": direction,
            "frame": frame,
            "source": {"reference_object": {"special_reference": frame}},
        }
    }
    direction_attribute = get_attribute(agent_memory, direction_lf)

    clause_text = "that is to {} {}".format(
        {"AGENT": "your", "SPEAKER": "my"}[frame], direction.lower()
    )
    clause_executable = {
        "input_left": {"attribute": direction_attribute},
        "comparison_type": "GREATER_THAN",
        "input_right": "0",
    }
    clause_lf = {
        "input_left": {"attribute": direction_lf},
        "comparison_type": "GREATER_THAN",
        "input_right": "0",
    }

    return clause_executable, clause_lf, clause_text, None


# def build_relative_spatial_clause(data):
#    """x > name_5.x, distance to (0,0,0) > (distance name_5 to (0,0,0), etc."""

# def build_extremal_spatial_clause(data):
#    """largest x, largest distance to (0,0,0) etc."""


def build_name_clause(data, agent_memory, opts):
    selectable_names = [
        x
        for x in data["names"]
        if ((x not in ["agent", "speaker", "mob"]) and (x[0] != "_"))
    ]
    name = selectable_names[torch.randint(len(selectable_names), (1,))].lower()
    clause = {"pred_text": "has_name", "obj_text": name}
    clause_text = "that is named " + name
    clause_triple_executable = {
        "AND": [
            {
                "input_left": {"attribute": "pred_text"},
                "comparison_type": "EQUAL",
                "input_right": "has_name",
            },
            {
                "input_left": {"attribute": "obj_text"},
                "comparison_type": "EQUAL",
                "input_right": name,
            },
        ]
    }
    return clause, deepcopy(clause), clause_text, clause_triple_executable


def build_tag_clause(data, agent_memory, opts):
    # don't use properties that start with "_"
    selectable_props = [
        x for x in data["props"] if ((x[0] != "_") and (x not in ["AGENT", "SELF"]))
    ]
    selectable_props = [
        x for x in selectable_props if (x not in ["hole", "mob", "voxel_object"])
    ]
    prop = selectable_props[torch.randint(len(selectable_props), (1,))].lower()
    clause = {"pred_text": "has_tag", "obj_text": prop}
    clause_text = "that has the property " + prop
    clause_triple_executable = {
        "AND": [
            {
                "input_left": {"attribute": "pred_text"},
                "comparison_type": "EQUAL",
                "input_right": "has_tag",
            },
            {
                "input_left": {"attribute": "obj_text"},
                "comparison_type": "EQUAL",
                "input_right": prop,
            },
        ]
    }
    return clause, deepcopy(clause), clause_text, clause_triple_executable


def get_clause_map(opts, configs):
    C = configs["clause_types"]

    clause_map = {
        "name": build_name_clause,
        "tag": build_tag_clause,
        "absolute_cardinal": build_absolute_cardinal_clause,
        "absolute_distance": build_absolute_distance_clause,
        "direction": build_direction_clause,
    }
    clause_types = {k: float(v) for k, v in C.items()}
    clause_opts = {
        k: configs.get("clause_opts", {}).get(k, {}) for k in clause_types.keys()
    }

    operator_probs = configs["operator_probs"]  # AND, OR, NOT

    return clause_map, clause_types, clause_opts, operator_probs


def build_q(query_data, agent_memory):
    """
    Builds the query in 4 different types:
    w[0]: executable query (for the memory.basic_search())
    w[1]: logical form query
    w[2]: text form query
    w[3]: executable query for finding the relevant triples. This needs
    #     to be done because the original executable query only returns the
    #     relevant reference objects. And we need the triples for triple supervision.
    #     Some query types don't have relevant triples, so they return None in w[3]
    """
    clauses = []
    sample_clause_types = []
    for i in range(query_data["num_clauses"]):
        cprobs = query_data["clause_data"]["clause_types"]
        clause_types = query_data["clause_data"]["clause_map"]
        clause_type = k_multinomial(cprobs)
        clause_builder = clause_types[clause_type]
        c = clause_builder(query_data["clause_data"], agent_memory, query_data["opts"])
        clauses.append(c)
        sample_clause_types.append(clause_type)

    operator_probs = query_data["clause_data"]["operator_probs"]
    sample_conjunction_type = "NONE"  # only works if there is <= 1 conjunction
    w = list(clauses.pop())
    while clauses:
        if torch.rand(1).item() < operator_probs["NOT"]:
            w = [
                {"NOT": [w[0]]},
                {"NOT": [w[1]]},
                "not " + w[2],
                {"NOT": [w[3]]} if w[3] else None,
            ]
        if torch.rand(1).item() < operator_probs["AND"]:
            sample_conjunction_type = "AND"
            c = clauses.pop()
            w = [
                {"AND": [w[0], c[0]]},
                {"AND": [w[1], c[1]]},
                w[2] + " and " + c[2],
                {"AND": [w[3], c[3]]} if w[3] and c[3] else (w[3] or c[3]),
            ]
        else:
            sample_conjunction_type = "OR"
            c = clauses.pop()
            w = [
                {"OR": [w[0], c[0]]},
                {"OR": [w[1], c[1]]},
                w[2] + " or " + c[2],
                {"OR": [w[3], c[3]]} if w[3] and c[3] else (w[3] or c[3]),
            ]

    if not (w[0].get("AND") or w[0].get("OR") or w[0].get("NOT")):
        w[0] = {"AND": [w[0]]}
        w[1] = {"AND": [w[1]]}

        if w[3]:
            w[3] = {"AND": [w[3]]}

    not_location_clause = {"pred_text": "has_tag", "obj_text": "_not_location"}
    w[0] = {
        "where_clause": {"AND": [w[0], not_location_clause]},
        "memory_type": "ReferenceObject",
    }
    w[1] = {
        "where_clause": {"AND": [w[1], not_location_clause]},
        "memory_type": "ReferenceObject",
    }
    if w[3]:
        w[3] = {"where_clause": w[3], "memory_type": "Triple"}

    return w, (sample_clause_types, sample_conjunction_type)


def build_simple_prop_output_clause(configs):
    """
    builds the output clause of the FILTERS lf
    outputs the clause, and text to prepend the query
    """
    p = configs.get("output_prop_probs", {"MEMORY": 1.0})
    prop = k_multinomial({x: float(p[x]) for x in p})
    if prop == "MEMORY":
        text = " what is the memory of the object ".format(prop)
        return "MEMORY", ""
    elif prop == "COUNT":
        return "COUNT", " what is the count of the object "
    elif prop == "LOCATION":
        return (
            [{"attribute": "x"}, {"attribute": "y"}, {"attribute": "z"}],
            " what is the location of the object ",
        )
    else:
        text = " what is the {} of the object ".format(prop.replace("has_tag", "tag"))
        return {"attribute": prop}, text


def reformat_list(input_list):
    int_list = [str(int(x)) if type(x) is float else x for x in input_list]
    int_str = "( {} )".format(" , ".join(int_list))
    return int_str


class FiltersQA(QA):
    """
    QA object corresponding to some basic FILTERS queries with logical forms
    """

    def __init__(self, data, agent=None, memory=None, snapshots=None):
        super().__init__(data, agent=agent, memory=memory)
        opts = data["opts"]
        configs = data["config"]
        if opts.fix_clauses:
            num_clauses = opts.num_clauses
        else:
            if opts.num_clauses == 1 or torch.rand(1).item() < opts.single_clause_prob:
                num_clauses = 1
            else:
                num_clauses = torch.randint(1, opts.num_clauses, (1,)).item() + 1

        clause_map, clause_types, clause_opts, operator_probs = get_clause_map(
            opts, configs
        )

        opts.clause_opts_from_config = clause_opts
        question_data = {}
        question_data["clause_map"] = clause_map
        question_data["clause_types"] = clause_types
        question_data["operator_probs"] = operator_probs
        u = get_universe_from_memory(memory)
        question_data["names"] = u["names"]
        question_data["props"] = u["props"]
        question_struct = {
            "num_clauses": num_clauses,
            "opts": opts,
            "clause_data": question_data,
        }
        (query, query_lf, query_text, query_triple_ex), (
            sample_clause_types,
            sample_conjunction_type,
        ) = build_q(question_struct, memory)
        output_lf, output_text_prepend = build_simple_prop_output_clause(configs)
        query["output"] = output_lf
        query_lf["output"] = output_lf
        query_text = (
            output_text_prepend + query_text + "?"
        )  # leading space for GPTTokenizer
        self.query = query
        self.question_lf = query_lf
        self.question_text = query_text
        self.query_triple_ex = query_triple_ex
        self.sample_text = False
        self.sample_clause_types = sample_clause_types
        self.sample_conjunction_type = sample_conjunction_type
        self.get_answer()

    def get_answer(self, query=None):
        query = query or self.query
        query_triple = self.query_triple_ex
        output = query["output"]

        # memids, _ = self.memory.basic_search("SELECT MEMORIES FROM Set WHERE (has_name=*)")
        # memids, vals = self.memory.basic_search("SELECT (x, y) FROM ReferenceObject",get_all=True)
        memids, vals = self.memory.basic_search(query, get_all=True)

        if query_triple:
            triple_memids, triple_vals = self.memory.basic_search(query_triple)
        else:
            triple_memids, triple_vals = [], []
        if not memids:
            self.memids_and_vals = ([NULL_MEMID], None)
            self.triple_memids_and_vals = ([NULL_MEMID], None)
            self.answer_text = "none"
            if type(output) is str and output == "COUNT":
                self.answer_text = " 0"
        else:
            self.memids_and_vals = (memids, vals)
            self.triple_memids_and_vals = (triple_memids, triple_vals)
            if type(output) is str and output == "MEMORY":
                self.answer_text = "mem"
            elif type(output) is str and output == "COUNT":
                self.answer_text = " " + str(len(memids))
            else:
                self.answer_text = ""
                vals = [
                    x for x in vals if (x is not None and x[0] != "_")
                ]  # remove non-queriable tokens

                if not vals:
                    self.answer_text = self.answer_text + " null"
                elif self.sample_text is True:
                    # randomly sample one token
                    val_choice = choice(vals)
                    if type(val_choice) is list:
                        val_choice = reformat_list(val_choice)
                    self.answer_text = self.answer_text + " " + str(val_choice)
                else:
                    for val in vals:
                        if type(val) is list:
                            val = reformat_list(val)
                        self.answer_text = self.answer_text + " " + str(val).lower()

        self.answer_text = self.answer_text.strip()

        if output != "LOCATION":
            # sort alphabetically since we are predicting a set sequentially
            self.answer_text = sorted(self.answer_text.split())
            self.answer_text = " ".join(self.answer_text)
        self.answer_text = "<|endoftext|>{}<|endoftext|>".format(self.answer_text)
