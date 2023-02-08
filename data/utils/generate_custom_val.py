"""
Generating custom validation set
"""

import torch
import os
import sys
from droidlet.memory.filters_conversions import sqly_to_new_filters
from nsm_utils import choice, k_multinomial, get_attribute, NULL_MEMID
from memory_utils import build_memory, convert_memory_simple, add_snapshot, get_memory_text
from prettytable import PrettyTable
from transformers import GPT2Tokenizer
from db_encoder import DBEncoder
from filters_based_queries import FiltersQA, build_simple_prop_output_clause, build_direction_clause, build_absolute_distance_clause, build_absolute_cardinal_clause
from geometric_minimax_queries import closest_farthest_object, closest_farthest_from_loc, max_direction, distance_between
from geometric_basic_queries import what_is_location_nearby, look_direction, where_is_object
from temporal_snapshot_queries import distance_moved, farthest_moved_object, location_at_time, farthest_direction_moved_object, direction_moved_object, get_output_from_mem
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
import math
import copy

SL = 15

"""
+---+--------------------------+----------------------+----------------+----------------------------------------+-------+------+
| T | RefObj                   | Location             | Pitch, Yaw     | Properties                             | Label | Pred |
+---+--------------------------+----------------------+----------------+----------------------------------------+-------+------+
| 0 | null                     | (0.00, 0.00, 0.00)   | ( 0.00,  0.00) |                                        |       |      |
| 0 | fake _ agent self self   | (11.00, 7.00, 14.00) | ( 0.20, -1.92) | ['agent', 'self']                      |       |      |
| 0 | aldo r abbit mob mob     | (6.00, 7.00, 14.00)  | (-0.74,  1.74) | ['rabbit', 'white', 'aldo']            |       |      |
| 0 | har ley ch icken mob mob | (8.00, 7.00, 11.00)  | (-1.28,  0.32) | ['chicken', 'gray', 'harley']          |       |      |
| 0 | h umph rey p ig mob mob  | (4.00, 7.00, 8.00)   | (-1.19,  0.57) | ['pig', 'gray', 'humphrey']            |       |      |
| 0 | s asha p ig mob mob      | (5.00, 7.00, 7.00)   | (-1.17,  3.12) | ['pig', 'gray', 'sasha']               |       |      |
| 0 | spe aker player player   | (13.00, 7.00, 4.00)  | (-0.61,  1.00) | ['speaker']                            |   *   |      |
| 0 | inst_seg inst_seg        | (1.00, 6.00, 2.00)   | ( 0.00,  0.00) | ['hole']                               |       |      |
| 1 | null                     | (0.00, 0.00, 0.00)   | ( 0.00,  0.00) |                                        |       |      |
| 1 | fake _ agent self self   | (11.00, 7.00, 14.00) | ( 0.20, -1.92) | ['agent', 'self']                      |       |      |
| 1 | aldo r abbit mob mob     | (8.00, 7.00, 9.00)   | (-0.74,  1.74) | ['rabbit', 'white', 'aldo']            |       |      |
| 1 | har ley ch icken mob mob | (13.00, 7.00, 8.00)  | (-1.28,  0.32) | ['chicken', 'gray', 'harley']          |       |      |
| 1 | h umph rey p ig mob mob  | (2.00, 7.00, 6.00)   | (-1.19,  0.57) | ['pig', 'gray', 'humphrey']            |       |      |
| 1 | s asha p ig mob mob      | (6.00, 7.00, 5.00)   | (-1.17,  3.12) | ['pig', 'gray', 'sasha']               |       |      |
| 1 | spe aker player player   | (13.00, 7.00, 4.00)  | (-0.61,  1.00) | ['speaker', 'attended_while_interpre'] |   *   |      |
| 1 | inst_seg inst_seg        | (1.00, 6.00, 2.00)   | ( 0.00,  0.00) | ['hole']                               |       |      |
+---+--------------------------+----------------------+----------------+----------------------------------------+-------+------+
"""

#FIXME a lot of this is duplicated code from other areas (build_q -> build_executable)

def name_to_memid(snapshots,name):
    for memid in snapshots["reference_objects"].keys():
        if snapshots["reference_objects"][memid][0]['name'] == name:
            return memid
    return False

def reformat_list(input_list):
    #FIXME this is repeated from filters_based_queries
    int_list = [str(int(x)) if type(x) is float else x for x in input_list]
    int_str = "( {} )".format(" , ".join(int_list))
    return int_str

# FIXME use QAZoo
def get_memids_and_text(memory, query, query_triple, output):
    print(query)
    output_lf, text_prefix = build_simple_prop_output_clause({"output_prop_probs":{"name": 1.0}})
    query["output"] = output_lf
    memids, vals = memory.basic_search(query, get_all=True)
    sample_text=False
    if query_triple is not None:
        triple_memids, triple_vals = memory.basic_search(query_triple)
    else:
        triple_memids, triple_vals = [],[]    
    if not memids:
        memids_and_vals = ([NULL_MEMID], None)
        triple_memids_and_vals = ([NULL_MEMID], None)
        answer_text = "none"
        if type(output) is str and output == "COUNT":
            answer_text = "0"
    else:                
        memids_and_vals = (memids, vals)
        triple_memids_and_vals = (triple_memids, triple_vals)
        if type(output) is str and output == "MEMORY":
            answer_text = "mem"
        elif type(output) is str and output == "COUNT":
            answer_text = " " + str(len(memids))
        else:
            answer_text = ""
            #FIXME if we return a list of lists (e.g. x,y,z), expand the sublist elements
            vals = [x for x in vals if (x is not None and x[0] != "_")] # remove non-queriable tokens #FIXME make consistent with mem
            
            if not vals:
                answer_text = answer_text + " null"
            elif sample_text is True:
                # randomly sample one token
                val_choice = choice(vals)
                if type(val_choice) is list:
                    val_choice = reformat_list(val_choice)
                answer_text = answer_text + " " + str(val_choice)
            else:
                for val in vals:
                    if type(val) is list:
                        val = reformat_list(val)
                    answer_text = answer_text + " " + str(val).lower()

    return memids_and_vals, triple_memids_and_vals, answer_text

def build_executable(clauses):
    operator_probs = {'AND':1.0, 'OR': 0.0, 'NOT': 0.0}
    sample_conjunction_type = "NONE" # only works if there is <= 1 conjunction
    w = list(clauses.pop())
    while clauses:
        if torch.rand(1).item() < operator_probs["NOT"]:
            w = [{"NOT": [w[0]]},
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
    w[0] = {"where_clause": {"AND": [w[0], not_location_clause]}, "memory_type": "ReferenceObject"}
    w[1] = {"where_clause": {"AND": [w[1], not_location_clause]}, "memory_type": "ReferenceObject"}
    if w[3]:
        w[3] = {"where_clause": w[3], "memory_type": "Triple"}

    return w

def create_direction_query(memory,direction,frame,return_type):
    """
    Input:
        memory: AgentMemory
        ctype = "GREATER_THAN
        pos = [12, 5, 8]
        value = 1

    Returns:
        clause executable that can be input to memory.searcher()
    """
    _, text_prefix = build_simple_prop_output_clause({"output_prop_probs":{return_type: 1.0}})
    (clause_executable, clause_lf, text_suffix, _) = build_direction_clause(None, memory, None, direction,frame)
    (executable, query_lf, query_text, query_triple_ex) = build_executable([(clause_executable, clause_lf, text_suffix, _)])
    query_text = text_prefix + text_suffix + "?"

    return executable, query_text, None, None, build_direction_clause.__name__

def create_distance_query(memory,ctype,value,pos,return_type):
    """
    Input:
        memory: AgentMemory
        ctype = "GREATER_THAN
        pos = [12, 5, 8]
        value = 1

    Returns:
        clause executable that can be input to memory.searcher()
    """
    assert ctype in ['GREATER_THAN','LESS_THAN']
    assert type(pos) is list

    _, text_prefix = build_simple_prop_output_clause({"output_prop_probs":{return_type: 1.0}})
    clause_executable, clause_lf, text_suffix, _ = build_absolute_distance_clause(None, memory, None, ctype, pos, value)

    (executable, query_lf, query_text, query_triple_ex) = build_executable([(clause_executable, clause_lf, text_suffix, _)])
    
    query_text = text_prefix + text_suffix + "?"

    return executable, query_text, None, None, build_absolute_distance_clause.__name__

def create_tag_queries(tags,return_type):
    """
    Input:
        tag: pig, sheep, blue, yellow, etc
        return_type: name, tag, loc, count

    Returns:
        clause executable that can be input to memory.searcher()
    """
    tag = tags[0] #only use 1 for now

    query = "SELECT name FROM ReferenceObject WHERE (has_tag={})".format(tag)
    clause_executable = sqly_to_new_filters(query)

    _, text_prefix = build_simple_prop_output_clause({"output_prop_probs":{return_type: 1.0}})
    text_suffix = "that has the property " + tag
    query_text = text_prefix + text_suffix + "?"

    return clause_executable, query_text, None, None, build_simple_prop_output_clause.__name__


def create_cardinal_clause_query(memory, cardinal, ctype, value, return_type):
    assert ctype in ['GREATER_THAN','LESS_THAN']

    _, text_prefix = build_simple_prop_output_clause({"output_prop_probs":{return_type: 1.0}})

    clause_executable, clause_lf, text_suffix, _ = build_absolute_cardinal_clause(None, memory, None, cardinal, ctype, value)

    (executable, query_lf, query_text, query_triple_ex) = build_executable([(clause_executable, clause_lf, text_suffix, _)])
    
    query_text = text_prefix + text_suffix + "?"

    return executable, query_text, None, None, build_absolute_cardinal_clause.__name__

def create_location_query(snapshots,opts,frame):
    query_text, refobj_memids, answer_text = where_is_object(snapshots,opts,None,frame)
    return None, query_text, refobj_memids, answer_text, where_is_object.__name__

def create_look_direction_query(snapshots,frame):
    query_text, refobj_memids, answer_text = look_direction(snapshots,None,None,frame)
    return None, query_text, refobj_memids, answer_text, look_direction.__name__


#################### GEOMETRIC ##################
def create_closest_farthest_obj_query(snapshots, name, polarity):
    memid = name_to_memid(snapshots, name)
    query_text, refobj_memids, answer_text = closest_farthest_object(snapshots, None, None, memid, polarity)

    return None, query_text, refobj_memids, answer_text, closest_farthest_object.__name__

def create_closest_farthest_loc_query(snapshots, pos, polarity):
    query_text, refobj_memids, answer_text = closest_farthest_from_loc(snapshots, None, None, pos, polarity)

    return None, query_text, refobj_memids, answer_text, closest_farthest_from_loc.__name__

def create_max_dir_query(snapshots, dtype, coord, direction, frame):
    query_text, refobj_memids, answer_text = max_direction(snapshots, None, None, dtype, coord, direction, frame)

    return None, query_text, refobj_memids, answer_text, max_direction.__name__

def create_distance_between_query(snapshots, obj1, obj2):
    obj1 = name_to_memid(snapshots, obj1)
    obj2 = name_to_memid(snapshots, obj2)
    query_text, refobj_memids, answer_text = distance_between(snapshots, None, None, obj1, obj2)

    return None, query_text, refobj_memids, answer_text, distance_between.__name__

def create_location_nearby_query(snapshots, opts, direction, frame, steps):
    query_text, refobj_memids, answer_text = what_is_location_nearby(snapshots, opts, None, direction, frame, steps)

    return None, query_text, refobj_memids, answer_text, what_is_location_nearby.__name__



################# TEMPORAL #########################
def create_distance_moved_query(snapshots, name):
    obj = name_to_memid(snapshots, name)
    question_text, memid, last_time_slice, distance  = distance_moved(snapshots, None, obj)

    return None, question_text, memid, distance, distance_moved.__name__

def create_farthest_moved_object_query(snapshots):
    question_text, memid, last_time_slice, _ = farthest_moved_object(snapshots, None)
    return None, question_text, memid, None, farthest_moved_object.__name__

def create_location_at_time_query(snapshots, name, time):
    obj = name_to_memid(snapshots, name)
    question_text, memid, last_time_slice, _ = location_at_time(snapshots, None, obj, time, name, obj)

    return None, question_text, memid, None, location_at_time.__name__


def create_farthest_direction_moved_object_query(snapshots, dtype, c, direction, frame):
    question_text, memid, last_time_slice, _ = farthest_direction_moved_object(snapshots, None, dtype, c, direction, frame)

    return None, question_text, memid, None, farthest_direction_moved_object.__name__

def create_direction_moved_object_object_query(snapshots, dtype, c, direction, frame):
    question_text, memid, last_time_slice, _ = direction_moved_object(snapshots, None, dtype, c, direction, frame)

    return None, question_text, memid, None, direction_moved_object.__name__

def create_queries(opts, world_spec, memory, snapshots=None):
    # chicken_exec1 = {"pred_text": "has_tag", "obj_text": "sheep"}
    # search_exec1 = {"where_clause": chicken_exec1, "memory_type": "ReferenceObject", 'output': {'attribute': "name"}}

    torch.manual_seed(0) # for keeping the names the same. #FIXME
                            
    TAG_QUERIES = [
        create_tag_queries(["chicken"], "NAME"),
        create_tag_queries(["pig"], "NAME"),
        create_tag_queries(["sheep"], "NAME"),
        create_tag_queries(["gray"], "NAME"),
    ]

    DIRECTION_QUERIES = [
        create_direction_query(memory, "RIGHT", "AGENT", "NAME"),
        create_direction_query(memory, "BACK", "AGENT", "NAME"),
        create_direction_query(memory, "RIGHT", "SPEAKER", "NAME"),
        create_direction_query(memory, "LEFT", "SPEAKER", "NAME"),
        create_direction_query(memory, "FRONT", "SPEAKER", "NAME"),
    ]

    DISTANCE_QUERIES = [
        create_distance_query(memory, 'LESS_THAN', 2, [3,5,4], "NAME"),
        create_distance_query(memory, 'LESS_THAN', 2, [5,5,4], "NAME"),
        create_distance_query(memory, 'GREATER_THAN', 5, [4,5,7], "NAME"),
        create_distance_query(memory, 'GREATER_THAN', 7, [4,5,2], "NAME"),
    ]

    CARDINAL_DISTANCE_QUERIES = [
        create_cardinal_clause_query(memory, "x", 'LESS_THAN', 2, "NAME"),
        create_cardinal_clause_query(memory, "x", 'LESS_THAN', 2,  "NAME"),
        create_cardinal_clause_query(memory, "x", 'GREATER_THAN', 2, "NAME"),
        create_cardinal_clause_query(memory, "x", 'GREATER_THAN', 2, "NAME"),
    ]


    CLOSEST_FARTHEST_OBJ_QUERIES = [
        create_closest_farthest_obj_query(snapshots, "speaker", -1), 
        create_closest_farthest_obj_query(snapshots, "fake_agent", -1),
        create_closest_farthest_obj_query(snapshots, "speaker", 1), 
        create_closest_farthest_obj_query(snapshots, "fake_agent", 1),
    ]

    CLOSEST_FARTHEST_LOC_QUERIES = [
        create_closest_farthest_loc_query(snapshots, [4, 5, 7], -1),
        create_closest_farthest_loc_query(snapshots, [4, 5, 7], 1),
        create_closest_farthest_loc_query(snapshots, [3, 5, 8], -1),
        create_closest_farthest_loc_query(snapshots, [3, 5, 8], 1),
    ]

    MAX_DIR_QUERIES = [
        create_max_dir_query(snapshots, "relative", None, "LEFT", "my"),
        create_max_dir_query(snapshots, "relative", None, "LEFT", "my"),
        create_max_dir_query(snapshots, "relative", None, "RIGHT", "my"),
        create_max_dir_query(snapshots, "relative", None, "RIGHT", "my"),
        create_max_dir_query(snapshots, "relative", None, "LEFT", "your"),
        create_max_dir_query(snapshots, "relative", None, "LEFT", "your"),
        create_max_dir_query(snapshots, "relative", None, "RIGHT", "your"),
        create_max_dir_query(snapshots, "relative", None, "RIGHT", "your"),
        create_max_dir_query(snapshots, "cardinal", "x", -1, None), # decreased
        create_max_dir_query(snapshots, "cardinal", "z", -1, None),
        create_max_dir_query(snapshots, "cardinal", "x", 1, None), #increased
        create_max_dir_query(snapshots, "cardinal", "z", 1, None),
    ]
    
    DISTANCE_BETWEEN_QUERIES = [
        create_distance_between_query(snapshots, "fake_agent", "speaker"),
        create_distance_between_query(snapshots, "aldo", "humphrey"), 
        create_distance_between_query(snapshots, "sasha", "speaker"), 
    ]


    DISTANCE_MOVED_QUERIES = [
        create_distance_moved_query(snapshots, "fake_agent"),
        create_distance_moved_query(snapshots, "speaker")
    ]

    FARTHEST_MOVED_OBJECT_QUERIES = [
        create_farthest_moved_object_query(snapshots)
    ]

    LOCATION_AT_TIME_QUERIES = [
        create_location_at_time_query(snapshots, "fake_agent", "end"),
        create_location_at_time_query(snapshots, "speaker", "end"),
        create_location_at_time_query(snapshots, "aldo", "end"),
        create_location_at_time_query(snapshots, "sasha", "end"),
    ]

    FARTHEST_DIRECTION_MOVED_QUERIES = [
        create_farthest_direction_moved_object_query(snapshots, "relative", None, "LEFT", "my"),
        create_farthest_direction_moved_object_query(snapshots, "relative", None, "RIGHT", "my"),
        create_farthest_direction_moved_object_query(snapshots, "relative", None, "FRONT", "my"),
        create_farthest_direction_moved_object_query(snapshots, "relative", None, "BACK", "my"),
        create_farthest_direction_moved_object_query(snapshots, "relative", None, "LEFT", "your"),
        create_farthest_direction_moved_object_query(snapshots, "relative", None, "RIGHT", "your"),
        create_farthest_direction_moved_object_query(snapshots, "relative", None, "FRONT", "your"),
        create_farthest_direction_moved_object_query(snapshots, "relative", None, "BACK", "your"),
        create_farthest_direction_moved_object_query(snapshots, "cardinal", "x", -1, None), # decreased
        create_farthest_direction_moved_object_query(snapshots, "cardinal", "z", -1, None),
        create_farthest_direction_moved_object_query(snapshots, "cardinal", "x", 1, None), # increased
        create_farthest_direction_moved_object_query(snapshots, "cardinal", "z", 1, None),
    ]

    DIRECTION_MOVED_QUERIES = [
        create_direction_moved_object_object_query(snapshots, "relative", None, "LEFT", "my"),
        create_direction_moved_object_object_query(snapshots, "relative", None, "RIGHT", "my"),
        create_direction_moved_object_object_query(snapshots, "relative", None, "FRONT", "my"),
        create_direction_moved_object_object_query(snapshots, "relative", None, "BACK", "my"),
        create_direction_moved_object_object_query(snapshots, "relative", None, "LEFT", "your"),
        create_direction_moved_object_object_query(snapshots, "relative", None, "RIGHT", "your"),
        create_direction_moved_object_object_query(snapshots, "relative", None, "FRONT", "your"),
        create_direction_moved_object_object_query(snapshots, "cardinal", "x", -1, None), # decreased
        create_direction_moved_object_object_query(snapshots, "cardinal", "z", -1, None),
        create_direction_moved_object_object_query(snapshots, "cardinal", "x", 1, None), # increased
        create_direction_moved_object_object_query(snapshots, "cardinal", "z", 1, None),
    ]

    LOCATION_NEARBY_QUERIES = [
        create_location_nearby_query(snapshots, opts, "RIGHT", "me", 2),
        create_location_nearby_query(snapshots, opts, "LEFT", "me", 2),
        create_location_nearby_query(snapshots, opts, "FRONT", "me", 2),
        create_location_nearby_query(snapshots, opts, "BACK", "me", 2),

        create_location_nearby_query(snapshots, opts, "RIGHT", "me", 3),
        create_location_nearby_query(snapshots, opts, "LEFT", "me", 3),
        create_location_nearby_query(snapshots, opts, "FRONT", "me", 3),
        create_location_nearby_query(snapshots, opts, "BACK", "me", 3),

        create_location_nearby_query(snapshots, opts, "RIGHT", "you", 1),
        create_location_nearby_query(snapshots, opts, "LEFT", "you", 1),
        create_location_nearby_query(snapshots, opts, "FRONT", "you", 1),
        create_location_nearby_query(snapshots, opts, "BACK", "you", 1),

        create_location_nearby_query(snapshots, opts, "RIGHT", "you", 4),
        create_location_nearby_query(snapshots, opts, "LEFT", "you", 4),
        create_location_nearby_query(snapshots, opts, "FRONT", "you", 4),
        create_location_nearby_query(snapshots, opts, "BACK", "you", 4),
    ]

    LOCATION_QUERIES = [
        create_location_query(snapshots, opts, "you"),
        create_location_query(snapshots, opts, "me"),
    ]

    LOOK_DIRECTION_QUERIES = [
        create_look_direction_query(snapshots, "you"),
        create_look_direction_query(snapshots, "me"),
    ]

    queries = []
    queries += TAG_QUERIES + DIRECTION_QUERIES + DISTANCE_QUERIES + CARDINAL_DISTANCE_QUERIES
    queries += CLOSEST_FARTHEST_LOC_QUERIES + MAX_DIR_QUERIES + DISTANCE_BETWEEN_QUERIES + CLOSEST_FARTHEST_OBJ_QUERIES + LOCATION_NEARBY_QUERIES
    queries += DISTANCE_MOVED_QUERIES + FARTHEST_MOVED_OBJECT_QUERIES + FARTHEST_DIRECTION_MOVED_QUERIES + DIRECTION_MOVED_QUERIES + LOCATION_AT_TIME_QUERIES
    queries += LOCATION_QUERIES + LOOK_DIRECTION_QUERIES

    # queries = LOCATION_AT_TIME_QUERIES

    return queries

def format_refobjs(word_tokens,tokenizer):
    token_list = []
    for tok in word_tokens:
        tok = tokenizer.convert_ids_to_tokens([tok.item()])[0]
        if tok not in ['!','none']:
            tok = tok.replace(' ','')
            token_list.append(tok)

    
    raw_words = ' '.join(token_list)
    raw_words = raw_words.replace('inst _ se g','inst_seg')
    raw_words = raw_words.replace('inst se g','inst_seg')
    raw_words = raw_words.replace('!',' ')
    raw_words = raw_words.replace('          ','')
    return raw_words


def format_triples(word_tokens,tokenizer):
    raw_words = ' '.join(tokenizer.convert_ids_to_tokens(word_tokens))
    raw_words = raw_words.replace(' [PAD]','').replace('[PAD]','')
    raw_words = raw_words.replace(' _ ','_')
    raw_words = raw_words.replace('has_tag_','has_tag ')
    raw_words = raw_words.replace(' none','').replace('none ','')
    raw_words = raw_words.replace(' triple','').replace('triple ','')

    raw_words = raw_words.replace('!',' ').replace('None','')
    raw_words = raw_words.replace('has_name','').replace('has_tag','')
    raw_words = raw_words.replace('tri ple','').replace('mob','')
    raw_words = raw_words.replace(' ','')

    return raw_words


def format_tensor(tensor_in):
    tensor_out = [ '%.2f' % elem for elem in tensor_in ]
    return tensor_out

def format_location(tensor_in, SL):
    # COORD_SHIFT = -SL//2
    # tensor_in[tensor_in!=0] = (tensor_in[tensor_in!=0]  * (SL / 2)) + (SL / 2 + COORD_SHIFT)
    xyz_max = SL-1
    xyz_min = 0
    # tensor_in[tensor_in!=0] = (((tensor_in[tensor_in!=0] + 1)/2)*(xyz_max-xyz_min)) + xyz_min
    tensor_in[tensor_in!=0] = (((tensor_in[tensor_in!=0]))*(xyz_max-xyz_min)) + xyz_min
    string_out = "({:.2f}, {:.2f}, {:.2f})".format(tensor_in[0],tensor_in[1],tensor_in[2])

    return string_out

def format_pitch_yaw(pitch, yaw):
    pitch = pitch.item()
    yaw = yaw.item()
    # pitch_min, pitch_max = -180, 180
    # yaw_min, yaw_max = -90, 90

    pitch_min, pitch_max = -math.pi, math.pi
    yaw_min, yaw_max = -math.pi, math.pi
    # pitch = (((pitch + 1)/2)*(pitch_max-pitch_min)) + pitch_min
    # yaw = (((yaw + 1)/2)*(yaw_max-yaw_min)) + yaw_min
    pitch = (((pitch))*(pitch_max-pitch_min)) + pitch_min
    yaw = (((yaw))*(yaw_max-yaw_min)) + yaw_min
    string_out = "({: .2f}, {: .2f})".format(pitch,yaw)

    return string_out

def print_memory(X, k, tokenizer, t_size, r_size, out_memid=None):
    context_text = X["context"]["text"]
    context_text = tokenizer.decode(context_text)
    context_text = context_text.replace('!','') #pad tokens in GPTtokenizer
    print('\nText Memory: "{}"\n'.format(context_text))


    t = PrettyTable(['T','RefObj', 'Location', 'Pitch, Yaw', 'Properties', 'Label', 'Pred'])
    t.align = "l"
    t.align["Label"] = "c"

    

    for TIMESTEP in range(X["context"]["r_attn"].size(0)):
        ys_ref_obj_k = X["context"]["r_attn"][TIMESTEP,:,0]

        triples_hash = X["context"]["t_hash"][TIMESTEP]
        triples_words = X["context"]["t_word"][TIMESTEP]
        triples_float = X["context"]["t_float"][TIMESTEP]

        ref_obj_hash = X["context"]["r_hash"][TIMESTEP]
        ref_obj_words = X["context"]["r_word"][TIMESTEP]
        ref_obj_float = X["context"]["r_float"][TIMESTEP]

        # loop through each ref_obj
        for r_idx in range(0, ref_obj_hash.size(0)):
            r_label = ys_ref_obj_k[r_idx].item()
            r_pred = None

            r_hash = ref_obj_hash[r_idx][0].item()
            r_words = ref_obj_words[r_idx]
            r_float = ref_obj_float[r_idx]


            r_words = format_refobjs(r_words, tokenizer)
            r_float_xyx = r_float[0:3]
            r_float_xyx = format_location(r_float_xyx,15) # currently rescaling by 15 to get back to xyz range
            pitch_and_yaw = format_pitch_yaw(r_float[3], r_float[4])
            
            if r_idx == 0:
                r_words = 'null'
            
            row_list = [TIMESTEP,r_words, r_float_xyx, pitch_and_yaw]

            # get corresponding triple idxs
            triple_idxs = (triples_hash[:,1] == r_hash).nonzero()
            triple_list = []

            for t_idx in triple_idxs:
                t_idx = t_idx.item()

                t_hash = triples_hash[t_idx]
                t_word = triples_words[t_idx]
                t_float = triples_float[t_idx]

                # if not a null ref_obj
                if t_hash.sum() > 0:
                    t_hash = format_tensor(t_hash)
                    t_float = format_tensor(t_float)
                    t_word = format_triples(t_word, tokenizer)
                    if t_word and t_word[0] != '_':
                        triple_list.append(t_word)

            
            row_list.append(triple_list) if triple_list else row_list.append('') 
            row_list.append('*') if r_label == 1 else row_list.append('')
            row_list.append(r_pred) if r_pred is not None else row_list.append('')
            
            # only add to print if not empty refobj
            if r_words.replace(" ","") != "": 
                t.add_row(row_list)

    print(t)

    return 



def create_data(opts, configs, world_spec, queries_list, memory, snapshots, encoder):
    sample_list = []
    for (query_executable, query_text, refobj_memids, answer_text, query_type) in queries_list:
        torch.manual_seed(42) # for keeping the names the same. #FIXME
        if query_executable:
            memids_and_vals, triple_memids_and_vals, answer_text = get_memids_and_text(memory, query_executable, None, "name")
        else:
            memids_and_vals = (refobj_memids, None)
            triple_memids_and_vals = ([NULL_MEMID], None)        

        if memids_and_vals[0] is None:
            memids_and_vals = ([NULL_MEMID], None)

        query_text = query_text.strip()
        query_text = " "+query_text
        
        if answer_text is None:
            try:
                # temporal queries need this extra step
                if type(refobj_memids) is not list:
                    refobj_memids_list = [refobj_memids]
                else:
                    refobj_memids_list = refobj_memids
                
                if any([c in query_text for c in ["where is", "where am", "where are", "location"]]):
                    print('*************')
                    output_type = "location"
                else:
                    output_type = "name"
                answer_text = get_output_from_mem(refobj_memids_list, snapshots, output_type, time_slice=1)
            except:
                answer_text = 'none'

        answer_text = answer_text.strip()
        query_text = "{}".format(query_text.lower())

        if not any([c in query_text for c in ["where is", "where am", "where are", "location"]]):
            answer_text = sorted(answer_text.split())
            answer_text = " ".join(answer_text)
        if answer_text in ['None']:
            answer_text = 'none'
        answer_text = "<|endoftext|>{}<|endoftext|>".format(answer_text)
        db_dump = {
            "question_text": query_text,
            "question_logical_form": "",
            "answer_text": answer_text,
            "memids_and_vals": memids_and_vals,
            "triple_memids_and_vals": triple_memids_and_vals,
            "sample_clause_types": [query_type],
            "sample_conjunction_type": [],
            "context": snapshots,
            "sample_query_type": query_type
        }

        snapshots["text"] = get_memory_text(memory, db_dump)


        encoded_sample = encoder.encode_all(db_dump)
        sample_list.append(encoded_sample)

        print_memory(copy.deepcopy(encoded_sample),0,tokenizer,None,None)
        print(query_text)
        print(answer_text)
        

    return sample_list


def get_text(tokens, tokenizer):
    text = tokenizer.convert_ids_to_tokens(tokens)    
    text = [x.replace('Ġ','') for x in text]
    return text

def format_words(word_tokens,tokenizer):
    raw_words = ' '.join(tokenizer.convert_ids_to_tokens(word_tokens))
    raw_words = raw_words.replace(' [PAD]','').replace('[PAD]','')
    raw_words = raw_words.replace(' _ ','_')
    raw_words = raw_words.replace('has_tag_','has_tag ')
    raw_words = raw_words.replace(' none','').replace('none ','')
    raw_words = raw_words.replace(' triple','').replace('triple ','')

    return raw_words


def format_tensor(tensor_in):
    tensor_out = [ '%.2f' % elem for elem in tensor_in ]
    return tensor_out
    
def format_location(tensor_in, SL):
    # COORD_SHIFT = -SL//2
    # tensor_in[tensor_in!=0] = (tensor_in[tensor_in!=0]  * (SL / 2)) + (SL / 2 + COORD_SHIFT)
    xyz_max = SL-1
    xyz_min = 0
    # tensor_in[tensor_in!=0] = (((tensor_in[tensor_in!=0] + 1)/2)*(xyz_max-xyz_min)) + xyz_min
    tensor_in[tensor_in!=0] = (((tensor_in[tensor_in!=0]))*(xyz_max-xyz_min)) + xyz_min
    string_out = "({:.2f}, {:.2f}, {:.2f})".format(tensor_in[0],tensor_in[1],tensor_in[2])

    return string_out
        

def generate_custom_val(opts, configs, world_spec, snapshots, save_dir, memory, encoder):        
    query_list = create_queries(opts, world_spec, memory, snapshots)

    data_list = create_data(opts, configs, world_spec, query_list, memory, snapshots, encoder)

    save_file = os.path.join(save_dir,'custom_test.pth')
    print('saving to {}'.format(save_file))
    torch.save(data_list,save_file)

    exit()

if __name__ == "__main__":
    generate_custom_val()