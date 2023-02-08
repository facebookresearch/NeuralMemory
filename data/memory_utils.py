import torch
from math import radians
from droidlet.base_util import Pos, Look
from droidlet.shared_data_struct.craftassist_shared_utils import Player, Item
from droidlet.memory.memory_nodes import (
    PlayerNode,
)
from droidlet.shared_data_struct import rotation
from droidlet.memory.craftassist.mc_memory_nodes import InstSegNode, MobNode
from droidlet.memory.craftassist.mc_memory import MCAgentMemory
from droidlet.shared_data_struct.craftassist_shared_utils import MOBS_BY_ID
from utils.mob_names import MOB_NAMES
from nsm_utils import AGENT_EID, SPEAKER_EID

NUM_MOB_NAMES = len(MOB_NAMES)
IDS_BY_MOB = {v: k for k, v in MOBS_BY_ID.items()}


class MobStruct:
    def __init__(self, x, y, z, pitch, yaw, eid, mobtype):
        self.pos = Pos(x, y, z)
        self.look = Look(yaw, pitch)
        self.entityId = eid
        self.mobType = IDS_BY_MOB[mobtype]


def random_location(opts, ground_height=0):
    if type(opts) is dict:
        SL = opts["SL"]
        flat = opts.get("flat_world", False)
    else:
        SL = opts.SL
        flat = getattr(opts, "flat_world", False)
    # location = torch.randint(-opts.SL//2, opts.SL//2, (3,)).tolist()
    location = torch.randint(0, SL, (3,)).tolist()
    if flat:
        location[1] = ground_height
    return location


def make_player_struct_from_spec(s, eid, name, item=None):
    # WARNING look is flipped in small scenes with shapes!!!!
    pitch, yaw = s["look"]
    if item:
        i = Item(*item)
    else:
        i = Item(0, 0)
    struct = Player(eid, name, Pos(*s["pos"]), Look(radians(yaw), radians(pitch)), i)
    return struct


def populate_memory_from_world_spec(spec, memory, use_inst_segs):
    self_player_struct = make_player_struct_from_spec(
        spec["agentInfo"], AGENT_EID, "agent"
    )
    PlayerNode.create(memory, self_player_struct, memid=memory.self_memid)
    speaker_struct = make_player_struct_from_spec(
        spec["avatarInfo"], SPEAKER_EID, "speaker"
    )
    PlayerNode.create(memory, speaker_struct)
    count = 1000

    for mob in spec["mobs"]:
        x, y, z, p, yaw = mob["pose"]
        count = count + 1
        mob_struct = MobStruct(x, y, z, radians(p), radians(yaw), count, mob["mobtype"])
        memid = MobNode.create(memory, mob_struct)
        memory.tag(subj_memid=memid, tag_text=mob["color"])
        name = MOB_NAMES[int(torch.randint(NUM_MOB_NAMES, (1,)))]
        memory.add_triple(subj=memid, pred_text="has_name", obj_text=name.lower())
        memory._db_write("UPDATE ReferenceObjects SET name=? WHERE uuid=?", name, memid)

    ps = memory._db_read("SELECT pitch FROM ReferenceObjects")
    if use_inst_segs:
        for inst_seg in spec["inst_seg_tags"]:
            if inst_seg.get("locs"):
                tags = [str(t).lower() for t in inst_seg["tags"] if t[0] != "_"]
                InstSegNode.create(memory, inst_seg["locs"], tags)

    return memory
    # TODO add names_and_properties/universe to memory?


def build_memory(opts, data=None, world_spec=None, use_inst_segs=True):
    memory = MCAgentMemory(coordinate_transforms=rotation)

    if world_spec:
        return populate_memory_from_world_spec(world_spec, memory, use_inst_segs)

    location = random_location(opts)
    self_player_struct = Player(AGENT_EID, "agent", Pos(*location), Look(0, 0))
    PlayerNode.create(memory, self_player_struct, memid=memory.self_memid)
    if data:
        ncount = 0
        names = {}
        for k, v in data["props"].items():
            if not names.get(k):
                location = data["locs"][k]
                names[k] = PlayerNode.create(
                    memory, Player(ncount + 1, k, Pos(*location), Look(0, 0))
                )
            else:
                memory.tag(subj_memid=names[k], tag_text=v)
        NAMES = list(data["props"].keys())
        PROPS = list(set(data["props"].values()))
    else:
        NAMES = ["name_{}".format(i) for i in range(opts.num_names)]
        PROPS = ["prop_{}".format(i) for i in range(opts.num_props)]
        for i in range(opts.num_names):
            location = random_location(opts)
            player_memid = PlayerNode.create(
                memory, Player(i + 1, NAMES[i], Pos(*location), Look(0, 0))
            )
            tagids = torch.randint(opts.num_props, (opts.props_per_name,)).tolist()
            props = set()
            for j in range(opts.props_per_name):
                memory.tag(subj_memid=player_memid, tag_text=PROPS[tagids[j]])
                props.add(PROPS[tagids[j]])

    names_and_properties = {"names": NAMES, "props": PROPS}
    memory.names_and_properties = names_and_properties

    return memory


def add_snapshot(snapshots, snapshot):
    if not snapshots.get("num_steps"):
        snapshots["num_steps"] = 0
    idx = snapshots["num_steps"]
    for key in ["triples", "reference_objects", "tasks"]:
        if not snapshots.get(key):
            snapshots[key] = {}
        content = snapshot.get(key, {})
        for e in content:
            if not snapshots[key].get(e["uuid"]):
                snapshots[key][e["uuid"]] = {idx: e}
            else:
                snapshots[key][e["uuid"]][idx] = e
    snapshots["num_steps"] += 1
    return snapshots


def add_basemem_info(agent_memory, record):
    memories_table_cmd = "SELECT * FROM Memories WHERE uuid=?"
    basemem_cols = agent_memory._db_read(memories_table_cmd, record["uuid"])
    record["node_type"] = basemem_cols[0][1]
    record["create_time"] = basemem_cols[0][2]
    record["updated_time"] = basemem_cols[0][3]
    record["attended_time"] = basemem_cols[0][3]
    return record


def convert_memory_simple(agent_memory, opts={}):
    """
    inputs agent memory, outputs a dict with possible fields
    "triples", "tasks", "reference_objects".
    each field contains a list of memories of the specified type
    each memory in the list is a dict with fields that depend on the
    type.
    all contain "node_type", "create_time", "updated_time", "attended_time"
    """
    out = {}
    if opts.get("all_triples"):
        memories_table_cmd = "SELECT * FROM Memories WHERE node_type !=?"
        memories_table = agent_memory._db_read(memories_table_cmd, "Schematic")
    else:
        memories_table_cmd = (
            "SELECT * FROM Memories WHERE" + " node_type !=? AND" * 4 + " node_type !=?"
        )
    triples_cmd = (
        "SELECT * FROM Triples Where subj IN ("
        + memories_table_cmd.replace("*", "uuid")
        + ")"
    )
    if opts.get("all_triples"):
        triples_table = agent_memory._db_read(triples_cmd, "Schematic")
    else:
        triples_table = agent_memory._db_read(
            triples_cmd,
            "Triple",
            "NamedAbstraction",
            "Schematic",
            "MobType",
            "BlockType",
        )
    columns = [
        "uuid",
        "subj",
        "subj_text",
        "pred",
        "pred_text",
        "obj",
        "obj_text",
        "confidence",
    ]
    out["triples"] = [
        add_basemem_info(agent_memory, dict(zip(columns, t))) for t in triples_table
    ]

    tasks_cmd = "SELECT uuid, action_name, paused, finished FROM Tasks"
    columns = ["uuid", "action_name", "paused", "finished"]
    tasks_table = agent_memory._db_read(tasks_cmd)
    out["tasks"] = [
        add_basemem_info(agent_memory, dict(zip(columns, t))) for t in tasks_table
    ]

    columns = [
        "uuid",
        "eid",
        "x",
        "y",
        "z",
        "yaw",
        "pitch",
        "name",
        "type_name",
        "ref_type",
        "player_placed",
        "voxel_count",
    ]
    ro_table = agent_memory._db_read(
        "SELECT " + ", ".join(columns) + " FROM ReferenceObjects"
    )
    out["reference_objects"] = [
        add_basemem_info(agent_memory, dict(zip(columns, t))) for t in ro_table
    ]
    for r in out["reference_objects"]:
        r["bounding_box"] = agent_memory.get_mem_by_id(r["uuid"]).get_bounds()
    return out


def get_memory_text(memory, snapshots):
    """"""
    # Creates the text form of the memories. This could be instead be done during the memory creation
    # This function returns a string where each sentence is a context memory.
    # E.g. "chris is at location (5, 4, 2). chris is a cow. chris is green. mary is at location ..."
    # NOTE: Will repeat many of the same facts for several time-steps if they haven't changed.
    ""
    if "context" in snapshots:
        context = snapshots["context"]
    else:
        context = snapshots

    refobj_ids = context["reference_objects"].keys()
    triple_ids = context["triples"].keys()
    memory_text_list = []
    for T in range(context["num_steps"]):
        for refobj_id in refobj_ids:
            if T in context["reference_objects"][refobj_id]:
                RO = context["reference_objects"][refobj_id][T]
                name = RO["name"]
                ref_type = RO["ref_type"]
                type_name = RO["type_name"]
                x = round(RO["x"])
                y = round(RO["y"])
                z = round(RO["z"])
                pitch = RO["pitch"] if RO["pitch"] is not None else ""
                yaw = RO["yaw"] if RO["yaw"] is not None else ""

                props = []
                for triple_id in triple_ids:
                    if T in context["triples"][triple_id]:
                        Tr = context["triples"][triple_id][T]
                        if Tr["subj"] == refobj_id:
                            prop = Tr["obj_text"]
                            props.append(prop)

                if not name and ref_type == "inst_seg":
                    name = ref_type

                if not name and ref_type == "item_stack":
                    name = type_name

                if name:
                    # alice is at location (5, 6, 8)
                    name = name.lower()

                    memory_text = "{} is at location ({},{},{}). ".format(name, x, y, z)
                    if pitch != "" or yaw != "":
                        try:
                            memory_text += (
                                "{} has pitch {:.2f} and yaw {:.2f}. ".format(
                                    name, pitch, yaw
                                )
                            )
                        except:
                            import ipdb

                            ipdb.set_trace()

                    for prop in props:
                        if prop:
                            prop = prop.lower()
                            if prop in [
                                "_in_others_inventory",
                                "_on_ground",
                                "_in_inventory",
                            ] or (prop[0] != "_" and prop != name and prop != "mob"):
                                memory_text += "{} has property {}. ".format(name, prop)
                    memory_text_list.append(memory_text)

    memory_text = "".join(memory_text_list)[0:-1]
    return memory_text


def check_inactive(qa_name, qa_obj, db_dump):
    """
    In a 2 clause sample this method checks to see if 1 of the clauses is inactive (not neccesary for answering the question)
    NOTE: only works for 2 clauses

    Returns True if one of the clauses is inactive.
    Returns False if both clauses are used
    """
    if qa_name == "FiltersQA":
        for clause_idx in range(2):  # 2 clauses
            query = qa_obj.query.copy()
            clauses = query["where_clause"]["AND"][0]
            num_clauses = len(clauses.get("AND", []))
            if num_clauses > 1:
                clauses["AND"] = [clauses["AND"][clause_idx]]
                qa_obj.get_answer(query)
                db_dump2 = qa_obj.get_all()
                is_eq = db_dump["memids_and_vals"] == db_dump2["memids_and_vals"]
                if is_eq:
                    return True

    return False
