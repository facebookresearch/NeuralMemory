import numpy as np
import sqlite3
import os
import sys

from math import radians
from droidlet.lowlevel.minecraft.pyworld.world import World
from droidlet.lowlevel.minecraft.iglu_util import IGLU_BLOCK_MAP
from droidlet.memory.craftassist.mc_memory_nodes import InstSegNode
from droidlet.lowlevel.minecraft.pyworld.fake_mobs import SimpleMob, make_mob_opts
from droidlet.lowlevel.minecraft.small_scenes_with_shapes import build_shape_scene
from nsm_utils import (
    make_world_args_from_config,
    SPEAKER_EID,
    AGENT_EID,
    choice,
    k_multinomial,
    generate_items,
)
from memory_utils import (
    add_snapshot,
    convert_memory_simple,
    make_player_struct_from_spec,
)
from utils.mob_names import MOB_NAMES
from command_generator import get_command

NUM_MOB_NAMES = len(MOB_NAMES)

this_path = os.path.dirname(os.path.abspath(__file__))
### FIXME!!
fake_agent_path = this_path.strip("data") + "fairo/agents/craftassist/tests/"
sys.path.append(fake_agent_path)
from fake_agent import FakeAgent, FakePlayer
from recorder import Recorder

from droidlet.lowlevel.minecraft.mc_util import SPAWN_OBJECTS, fill_idmeta
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import (
    BORING_BLOCKS,
    PASSABLE_BLOCKS,
)
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import (
    COLOR_BID_MAP,
)
from droidlet.lowlevel.minecraft import craftassist_specs

# FIXME: getfrom config:
MAX_NUM_ITEMS = 6

# for debugging:
def seq_print(s):
    if type(s) is list:
        for e in s:
            print(e)
    elif type(s) is dict:
        for k, v in s.items():
            print("{}: {}".format(k, v))


class Opt:
    pass


def low_level_data():
    low_level_data = {
        "mobs": SPAWN_OBJECTS,
        "block_data": craftassist_specs.get_block_data(),
        "block_property_data": craftassist_specs.get_block_property_data(),
        "color_data": craftassist_specs.get_colour_data(),
        "boring_blocks": BORING_BLOCKS,
        "passable_blocks": PASSABLE_BLOCKS,
        "fill_idmeta": fill_idmeta,
        "color_bid_map": COLOR_BID_MAP,
    }
    return low_level_data


# FIXME do this in fairo proper, reuse in test cases and pyworld agent
class EpisodeRunner:
    def __init__(self, agent, snapshot_freq=0, keep_on_truckin=False):
        self.agent = agent
        self.agent.no_path_block_clearing = True
        self.snapshot_freq = snapshot_freq
        self.snapshots = {}
        # set this if we want to ignore agent action errors
        self.keep_on_truckin = keep_on_truckin

    def add_snapshot(self):
        snapshot = convert_memory_simple(self.agent.memory)
        self.snapshots = add_snapshot(self.snapshots, snapshot)
        # FIXME
        if self.store_blocks:
            if self.snapshots.get("blocks") is None:
                self.snapshots["blocks"] = []
            idx = np.nonzero(self.agent.world.blocks[:, :, :, 0])
            nz = self.agent.world.blocks[idx]
            idx = list(zip(*idx))
            B = [
                (int(x), int(y), int(z), int(l), int(m))
                for (x, y, z), (l, m) in zip(idx, nz)
            ]
            self.snapshots["blocks"].append(
                [(x, y, z, IGLU_BLOCK_MAP[(i, m)]) for x, y, z, i, m in B]
            )

    def run_episode(
        self,
        logical_form=None,
        speaker=None,
        chatstr="",
        answer=None,
        stop_on_chat=False,
        max_steps=1000,
        min_steps=0,
        no_agent_step=False,
        store_blocks=False,
    ):
        """run an episode

        If "answer" is specified and a question is asked by the agent, respond
        with this string.

        If "stop_on_chat" is specified, stop iterating if the agent says anything

        if snapshot_freq < 0, only makes a snapshot at end of episode
                         == 0 makes a snapshot at beginning and end
                         > 0 makes a snapshot every snapshot_freq steps and after the episode
        """
        self.store_blocks = store_blocks
        if logical_form is not None:
            chatstr = chatstr or "TEST {}".format(logical_form)
            speaker = speaker or list(self.agent.world.players.values())[0].name
            self.add_incoming_chat(chatstr, speaker)
            self.agent.set_logical_form(logical_form, chatstr, speaker)

        # FIXME, do this better (so there could be multiple sequential lfs, etc.).
        # its stored in memory...
        self.snapshots["lf"] = logical_form
        self.snapshots["chatstr"] = chatstr

        if self.snapshot_freq == 0:
            self.add_snapshot()
        status = self.flush(
            min_steps, max_steps, stop_on_chat=stop_on_chat, no_agent_step=no_agent_step
        )
        if logical_form is None:
            assert answer is None
        if answer is not None:
            self.add_incoming_chat(answer, speaker)
            self.flush(
                0, max_steps, stop_on_chat=stop_on_chat, no_agent_step=no_agent_step
            )
        self.add_snapshot()
        return status

    def flush(
        self, min_steps=0, max_steps=10000, stop_on_chat=False, no_agent_step=False
    ):
        """Run the agant's step until task and dialogue stacks are empty

        If "stop_on_chat" is specified, stop iterating if the agent says anything

        Return a flag to tell if epsiode ran till end
        """
        success = True
        if stop_on_chat:
            self.agent.clear_outgoing_chats()
        for i in range(max_steps):
            if self.snapshot_freq > 0:
                if i % self.snapshot_freq == 0:
                    self.add_snapshot()
            if no_agent_step:
                # FIXME make a dummy step
                self.agent.count = self.agent.count + 1
                # WARNING this will step the world faster than if agent was stepping..
                self.agent.world.step()
                # FIXME use agent's dummy step and proper perceive
                perception_output = self.agent.perception_modules["low_level"].perceive(
                    force=True
                )
                self.agent.memory.update(perception_output)
                if hasattr(self.agent, "recorder"):
                    self.agent.recorder.record_world()
            else:
                try:
                    self.agent.step()
                except Exception as e:
                    # import traceback
                    # print(traceback.format_exc())

                    if self.keep_on_truckin:
                        pass
                    else:
                        return False
                should_stop, success = self.agent_should_stop(stop_on_chat)
                if should_stop and i >= min_steps:
                    break
        return success

    #                if i == max_steps - 1:
    #                    print("warning in {} : agent ran till max_steps".format(self))

    def agent_should_stop(self, stop_on_chat=False):
        # success test is not careful...!
        stop = False
        success = True
        _, interpreter_mems = self.agent.memory.basic_search(
            "SELECT MEMORY FROM Interpreter WHERE finished = 0"
        )
        if len(interpreter_mems) == 0 and not self.agent.memory.task_stack_peek():
            stop = True

        # stuck waiting for answer?
        _, answer_task_mems = self.agent.memory.basic_search(
            "SELECT MEMORY FROM Task WHERE (action_name=awaitresponse AND prio>-1)"
        )
        if answer_task_mems and not any([m.finished for m in answer_task_mems]):
            stop = True
        if stop_on_chat and self.agent.get_last_outgoing_chat():
            stop = True
        r = self.agent.recorder.get_last_record()
        actions = r.get("actions")
        if actions:
            success = all([self.check_action_status(a) for a in actions])
            if not success:
                stop = True
        return stop, success

    def check_action_status(self, action_record):
        ok = True
        if action_record["name"] == "place_block" or action_record["name"] == "dig":
            # FIXME account for coordinate shift if necessary:
            if any([action_record["args"][i] < 0 for i in range(3)]):
                ok = False
            if any([action_record["args"][i] >= self.agent.world.sl for i in range(3)]):
                ok = False
        if action_record["name"] == "send_chat":
            if "giving up" in action_record["args"]:
                ok = False
        return ok

    def add_incoming_chat(self, chat: str, speaker_name: str, add_to_memory=False):
        """Add a chat to memory as if it was just spoken by SPEAKER"""
        self.agent.world.chat_log.append("<" + speaker_name + ">" + " " + chat)
        if add_to_memory:
            self.agent.memory.add_chat(
                self.agent.memory.get_player_by_name(self.speaker).memid, chat
            )


class WorldBuilder:
    def __init__(self):
        class NubWorld:
            def __init__(self):
                self.count = 0
                self.players = {}

            def new_eid(self):
                return AGENT_EID

        A = FakeAgent(NubWorld(), low_level_data=low_level_data())
        A.memory._db_write("DELETE FROM MEMORIES WHERE uuid=?", A.memory.self_memid)
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        A.memory.db.backup(self.db)

    def instantiate_world_from_spec(self, config, opts):
        scene_opts = make_world_args_from_config(config)
        scene_spec = build_shape_scene(scene_opts)

        def block_generator(world):
            world.blocks[:] = 0
            for b in scene_spec["schematic_for_cuberite"]:
                x, y, z = world.to_npy_coords((b["x"], b["y"], b["z"]))
                # TODO maybe don't just eat every error, do this more carefully
                try:
                    world.blocks[x, y, z][0] = b["id"]
                    world.blocks[x, y, z][1] = b["meta"]
                except:
                    pass

        world_opts = Opt()
        world_opts.sl = scene_opts.SL
        speaker_struct = make_player_struct_from_spec(
            scene_spec["avatarInfo"], SPEAKER_EID, "speaker"
        )
        P = FakePlayer(
            struct=speaker_struct,
            low_level_data=low_level_data(),
            prebuilt_db=self.db,
            use_place_field=False,
        )
        players = [P]
        mobs = []
        for mob_spec in scene_spec["mobs"]:
            # FIXME add more possibilities:
            assert mob_spec["mobtype"] in ["rabbit", "cow", "pig", "chicken", "sheep"]
            mob_opt = make_mob_opts(mob_spec["mobtype"])
            x, y, z, pitch, yaw = mob_spec["pose"]
            # grabby mob config should be of form:
            # "grabby_mob_config": "pick_prob:.5:drop_prob:.1:pick_range:2",
            if config.get("WORLD", {}).get("grabby_mob_config"):
                grabby = config.get("WORLD", {}).get("grabby_mob_config").split(";")
                for o in grabby:
                    k, v = o.split(":")
                    setattr(mob_opt, k, float(v))
            mobs.append(
                SimpleMob(
                    mob_opt,
                    start_pos=(x, y, z),
                    start_look=(radians(yaw), radians(pitch)),
                )
            )
        items = generate_items(np.random.randint(1, MAX_NUM_ITEMS))
        world = World(
            world_opts,
            {
                "ground_generator": block_generator,
                "mobs": mobs,
                "players": players,
                "agent": None,
                "items": items,
            },
        )
        # FIXME this is a bad way to init
        world.agent_data = {
            "pos": scene_spec["agentInfo"]["pos"],
            "look": None,
            "eid": AGENT_EID,
            "name": "agent",
        }
        A = FakeAgent(
            world,
            low_level_data=low_level_data(),
            prebuilt_db=self.db,
            use_place_field=False,
        )
        # FIXME look is flipped in small scenes with shapes!!!!
        # and also this is an awful way to set up, FIXME
        pitch, yaw = scene_spec["agentInfo"]["look"]
        A.set_look_vec(*A.coordinate_transforms.look_vec(radians(yaw), radians(pitch)))

        # these are tags of various reference objects that might be used in building a command for agent:
        refobjs_for_command = []

        # add some pickable/droppable items
        items = world.get_items()
        # maybe put an item with the players and/or agent:
        item_tags = {i: items[i]["typeName"] for i in range(len(items))}
        for p in [P, A]:
            if np.random.rand() > 0.5:
                try:
                    i = choice(list(item_tags.keys()))
                    item = items[i]
                    ieid = item["entityId"]
                    world.player_pick_drop_items(p.entityId, [ieid])
                    world.items[ieid].update_position(*p.pos)
                    item_tags.pop(i, None)
                except:
                    pass

        refobjs_for_command.extend(list(item_tags.values()))

        perception_output = A.perception_modules["low_level"].perceive(force=True)
        A.memory.update(perception_output)

        R = Recorder(agent=A)
        A.recorder = R

        memory = A.memory
        # FIXME!  this is not keeping the mobs from the spec associated with their colors!
        # need to record eids at mob creation
        colors = [m["color"] for m in scene_spec["mobs"]]
        mob_memids = [
            m[0]
            for m in memory._db_read(
                "SELECT uuid FROM Memories WHERE node_type=?", "Mob"
            )
        ]

        mobs = []
        for memid in mob_memids:
            # FIXME!
            color = choice(colors)
            memory.nodes["Triple"].tag(memory, subj_memid=memid, tag_text=color)
            name = choice(MOB_NAMES).lower()
            memory.nodes["Triple"].tag(memory, subj_memid=memid, tag_text=name)
            memory._db_write(
                "UPDATE ReferenceObjects SET name=? WHERE uuid=?", name, memid
            )
            refobjs_for_command.append(name)
            mobs.append(name)

        inst_segs = []
        for inst_seg in scene_spec["inst_seg_tags"]:
            if inst_seg.get("locs"):
                # FIXME in fairo the inst_seg tags have a weird numpy type instead of str
                tags = [str(t).lower() for t in inst_seg["tags"] if t[0] != "_"]
                # FIXME the blocks in the inst_seg might be outside the world boundary!
                memid = InstSegNode.create(memory, inst_seg["locs"], tags)
                t = choice(tags)
                refobjs_for_command.append(t)
                if t != "hole":
                    inst_segs.append(t.lower())

        obj_data = {
            "all_objs": refobjs_for_command,
            "mobs": mobs,
            "inst_segs": inst_segs,
            "items": items,
        }

        p = scene_opts.agent_lf_prob
        lf = None
        chat = ""
        if k_multinomial({True: p, False: 1 - p}):
            if getattr(opts, "command_probs", None) is not None:
                lf, chat = get_command(obj_data, command_probs=opts.command_probs)
            else:
                lf, chat = get_command(obj_data)

        return A, world, lf, chat


if __name__ == "__main__":
    from config_args import get_opts_and_config

    opts, configs = get_opts_and_config()
    N = 50
    W = WorldBuilder()
    a, w, lf, chat = W.instantiate_world_from_spec(configs, opts)
    runner = EpisodeRunner(a, snapshot_freq=0)
    if lf is not None:
        runner.run_episode(
            min_steps=N, max_steps=N, logical_form=lf, chatstr=chat, store_blocks=True
        )
    else:
        runner.run_episode(
            min_steps=N, max_steps=N, no_agent_step=True, store_blocks=True
        )
