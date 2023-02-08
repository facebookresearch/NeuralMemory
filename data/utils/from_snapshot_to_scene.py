import os


def get_pose(r):
    return r["x"], r["y"], r["z"], r["yaw"], r["pitch"]


def snapshot_to_scenes(s):
    scenes = []
    mobs = []
    for i in range(s["num_steps"]):
        scenes.append({})
        mobs.append([])
    for memid in s["reference_objects"]:
        for timestep in s["reference_objects"][memid]:
            r = s["reference_objects"][memid][timestep]
            x, y, z, yaw, pitch = get_pose(r)
            if r["name"] == "speaker":
                scenes[timestep]["avatarInfo"] = {
                    "pos": (x, y, z),
                    "look": (pitch, yaw),
                }
            elif r["name"] == "fake_agent":
                scenes[timestep]["agentInfo"] = {"pos": (x, y, z), "look": (pitch, yaw)}
            elif yaw is not None:
                # a mob
                mobs[timestep].append(
                    {"mobtype": r["type_name"], "pose": (x, y, z, pitch, yaw)}
                )
    for i in range(s["num_steps"]):
        scenes[i]["blocks"] = s["blocks"][i]
        scenes[i]["mobs"] = mobs[i]
    return scenes


if __name__ == "__main__":
    from config_args import get_opts_and_config
    from active_agent_world import WorldBuilder, EpisodeRunner
    from query_builder import QAZoo
    from memory_utils import get_memory_text
    import sys, os
    import torch
    import json
    from nsm_utils import NULL_MEMID

    this_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(this_path)
    transformemnn_path = this_path.strip("data") + "training/"
    sys.path.append(transformemnn_path + "db_dataloader")
    droidlet_path = this_path.strip("data") + "fairo/"
    sys.path.append(droidlet_path)
    sys.path.append(transformemnn_path + "utils")
    from db_encoder import DBEncoder

    encoded_samples = []
    snapshots_list = []

    opts, configs = get_opts_and_config()

    username = os.getlogin()
    root_dir = "/checkpoint/" + username + "/snapshots/"

    os.makedirs(root_dir, exist_ok=True)

    sample_idx = 0
    for idx in range(10):
        print(idx)
        N = 50
        W = WorldBuilder()
        a, w, lf, chat = W.instantiate_world_from_spec(configs, opts)
        runner = EpisodeRunner(a, snapshot_freq=0)
        if lf is not None:
            runner.run_episode(
                min_steps=N,
                max_steps=N,
                logical_form=lf,
                chatstr=chat,
                store_blocks=True,
            )
        else:
            runner.run_episode(
                min_steps=N, max_steps=N, no_agent_step=True, store_blocks=True
            )

        snapshots = runner.snapshots
        memory = runner.agent.memory

        J = snapshot_to_scenes(snapshots)

        encoder = DBEncoder(
            query_encode=opts.query_encode,
            SL=opts.SL,
            COORD_SHIFT=(-opts.SL // 2, -opts.SL // 2, -opts.SL // 2),
            tokenizer=opts.tokenizer,
            tokenizer_path=opts.tokenizer_path,
            remove_instseg_locations=True,
        )

        use_sample = True
        try:
            Z = QAZoo(opts, configs)
            db_dump = Z.get_qa(memory=memory, snapshots=snapshots)
            # if not ('increased' in db_dump['question_text']): # or 'farthest' in db_dump['question_text']):
            #     use_sample=False

            # if (not 'property' in db_dump['question_text']) or ('not' in db_dump['question_text']): # or 'farthest' in db_dump['question_text']):
            #     use_sample=False

            if "none" in db_dump["answer_text"].lower():
                use_sample = False

            if (db_dump is False) or (db_dump["memids_and_vals"][0] == NULL_MEMID):
                use_sample = False

        except TypeError:
            use_sample = False

        if use_sample:
            db_dump["context"]["text"] = get_memory_text(memory, snapshots)
            encoded_sample = encoder.encode_all(db_dump)
            encoded_samples.append(encoded_sample)
            snapshots_list.append(J)
            json_object = json.dumps(J, indent=4)

            with open(
                "/checkpoint/"
                + user
                + "/snapshots/samples"
                + str(sample_idx)
                + ".json",
                "w",
            ) as outfile:
                outfile.write(json_object)

            sample_idx += 1

    print("saving to : {}".format(root_dir + "encoded_samples.pth"))
    torch.save(encoded_samples, root_dir + "encoded_samples.pth")
    torch.save(snapshots_list, root_dir + "snapshots.pth")
