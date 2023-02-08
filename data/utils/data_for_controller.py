import os
import sys
from active_agent_world import WorldBuilder, EpisodeRunner
from config_args import get_opts_and_config
from memory_utils import add_snapshot, convert_memory_simple, get_memory_text
from nsm_utils import NULL_MEMID

this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_path)
transformemnn_path = this_path.strip("data") + "/training/"
sys.path.append(transformemnn_path + "/db_dataloader")
droidlet_path = this_path.strip("data") + "/fairo/"
sys.path.append(droidlet_path)
sys.path.append(transformemnn_path + "/utils/")
sys.path.append(transformemnn_path)
from db_encoder import DBEncoder


def extract_target(agent_memory, db_dump):
    _, task_mems = agent_memory.basic_search("SELECT MEMORY FROM Task")
    db_dump["memids_and_vals"] = ([], None)
    db_dump["question_logical_form"] = "NULL"
    db_dump["triple_memids_and_vals"] = ([NULL_MEMID], None)
    for task_mem in task_mems:
        if task_mem.action_name == "move":
            # TODO add the refobj_memid if its there
            db_dump["answer_text"] = "move " + str(task_mem.task.target.tolist())
            return True
        elif task_mem.action_name == "get":
            db_dump["answer_text"] = "get"
            _, objs = agent_memory.basic_search(
                "SELECT MEMORY FROM ReferenceObject WHERE eid={}".format(
                    task_mem.task.eid
                )
            )
            # return memid of object to get
            if objs:
                db_dump["memids_and_vals"] = ([objs[0].memid], None)
            return True
    return False


if __name__ == "__main__":
    opts, configs = get_opts_and_config()
    command_probs = {
        "move_loc": 1.0,
        "give_get_bring": 1.0,
    }
    opts.command_probs = command_probs
    encoder = DBEncoder(
        query_encode=opts.query_encode,
        SL=opts.SL,
        COORD_SHIFT=(-opts.SL // 2, -opts.SL // 2, -opts.SL // 2),
        tokenizer=opts.tokenizer,
        tokenizer_path=opts.tokenizer_path,
        remove_instseg_locations=True,
    )
    W = WorldBuilder()

    examples = []
    for i in range(opts.num_examples):
        a, w, lf, chat = W.instantiate_world_from_spec(configs, opts)
        runner = EpisodeRunner(a, snapshot_freq=0)
        status = runner.run_episode(max_steps=1, logical_form=lf, chatstr=chat)
        db_dump = {
            "question_text": chat,  # using "question_text" field to hold the command
            "context": add_snapshot({}, convert_memory_simple(a.memory)),
            "sample_clause_types": "NULL",  # FIXME
        }
        db_dump["context"]["text"] = get_memory_text(a.memory, db_dump)

        is_valid_command = extract_target(a.memory, db_dump)
        if is_valid_command:
            encoded_example = encoder.encode_all(db_dump)
            examples.append(encoded_example)
