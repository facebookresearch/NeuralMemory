from active_agent_world import WorldBuilder, EpisodeRunner
from config_args import get_opts_and_config

if __name__ == "__main__":
    opts, configs = get_opts_and_config()
    N = 500
    W = WorldBuilder()
    for i in range(100):
        a, w, lf, chat = W.instantiate_world_from_spec(configs, opts)
        runner = EpisodeRunner(a, snapshot_freq=0)
        status = runner.run_episode(max_steps=N, logical_form=lf, chatstr=chat)
        print(chat)
        print(status)
    # you can save the recorder objects with
    # a.recorder.save_to_file(<path>);
    # you should only save the ones with status True.
