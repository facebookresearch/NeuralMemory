import os
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Logger(object):
    def __init__(self, args, batch_sz):
        self.args = args

        self.log_path = ""
        if args.plot and args.local_rank == 0:
            self.run_name = args.model_type

            if args.model_type == "transformer":
                self.run_name += (
                    ".layers_"
                    + str(args.enc_layers)
                    + ".nhead_"
                    + str(args.nheads)
                    + ".dim_"
                    + str(args.embedding_dim)
                )

            self.run_name += ".ct_" + str(args.context_type)
            self.run_name += ".it_" + str(args.input_type)
            self.run_name += (
                ".opt_"
                + str(args.optim)
                + ".lr_"
                + str(args.lr)
                + ".warm_"
                + str(args.lr_warmup)
                + ".drop_"
                + str(args.dropout)
                + ".bsz_"
                + str(batch_sz * args.accumulation_steps)
                + ".ep_"
                + str(args.nepochs)
            )

            if args.cosine_decay:
                self.run_name += ".cd"

            if args.text_loss:
                self.run_name += ".tl"
                if args.text_kl:
                    self.run_name += "_kl"

            elif args.memid_loss:
                self.run_name += ".ml"
            else:
                self.run_name += ".tml"

            if args.triple_supervision:
                self.run_name += ".ts"  # +str(args.triple_supervision_prob)

            if args.attn_supervision:
                self.run_name += (
                    ".as_"
                    + str(args.attn_supervision_type)
                    + "_"
                    + str(args.attn_supervision_weight)
                )

            if args.normalize_floats:
                self.run_name += ".fn"

            if args.run_name != "":
                self.run_name += "." + args.run_name

            self.run_name += ".seed_" + str(args.seed)

            if args.debug:
                self.run_name += ".debug"

            if ".pth" in args.simple_data_path:
                dataset = args.simple_data_path.split("/")[-2].replace(".pth", "")
            else:
                dataset = args.simple_data_path.split("/")[-1]

            self.log_path = os.path.join(
                args.plot_dir, dataset, self.run_name, args.plot_name
            )

            print("saving plots to: {}".format(self.log_path))
            if (
                os.path.exists(self.log_path)
                and not args.overwrite_logs
                and not args.full_test
                and not "debug" in self.run_name
            ):
                print("Experiment already exists. use --overwrite \nEXITING")
                exit(0)
            elif not os.path.exists(self.log_path):
                os.makedirs(self.log_path)

            self.writer = SummaryWriter(self.log_path)

            if args.wandb:
                import wandb

                # If you don't want your script to sync to the cloud
                os.environ["WANDB_MODE"] = "offline"
                # os.environ['WANDB_DIR'] = args.wandb_dir
                os.environ["WANDB_DIR"] = os.path.join(
                    args.plot_dir, dataset, self.run_name, args.plot_name
                )
                # os.environ['WANDB_RESUME'] = 'auto'

                self.wandb = wandb
                name_without_seed = self.run_name.split("seed_")[0] + ".".join(
                    self.run_name.split("seed_")[1].split(".")[1:]
                )
                wandb_config = {
                    "embedding_dim": args.embedding_dim,
                    "enc_layers": args.enc_layers,
                    "nhead_": args.nheads,
                    "lr_warmup": args.lr_warmup,
                    "dropout": args.dropout,
                    "input_type": args.input_type,
                    "optim": args.optim,
                    "lr": args.lr,
                    "batch_sz": (batch_sz * args.accumulation_steps),
                    "nepochs": args.nepochs,
                    "seed": args.seed,
                    "triple_supervision": args.triple_supervision,
                    "attn_supervision": args.attn_supervision,
                    "attn_supervision_type": args.attn_supervision_type,
                    "attn_supervision_weight": args.attn_supervision_weight,
                    "run_name": args.run_name,
                    "large_train": args.large_train,
                    "memid_loss": args.memid_loss,
                    "text_loss": args.text_loss,
                    "group": name_without_seed,
                    "context_type": args.context_type,
                    "pretrained_gpt": args.pretrained_gpt,
                    "dir": args.simple_data_path.split("/")[-1],
                }

                try:
                    self.wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        name=self.run_name,
                        config=wandb_config,
                        resume=True,
                    )
                except:
                    print("WANDB INIT FAILED")

            self.output_file_name = os.path.join(self.log_path, "log.txt")
            f = open(self.output_file_name, "w")
            f.close()

        self.logs = dict()
        self.best_total_loss = float("inf")
        self.best_text_loss = float("inf")
        self.best_text_err = float("inf")
        self.best_memid_loss = float("inf")
        self.best_memid_ref_obj_loss = float("inf")
        self.best_memid_KL = float("inf")

        self.loss_list = []

    def print(self, msg):
        print(msg)
        if self.args.plot:
            self.writer.add_text("stdout", msg)

    def set_state(self, state):
        self.logs = state

    def get_state(self):
        return self.logs

    def log(self, title, value):
        if title not in self.logs:
            self.logs[title] = {"data": [], "type": "line"}
        self.logs[title]["data"].append(value)

    def plot_step(self, step):
        for title, v in self.logs.items():
            if v["type"] == "line":
                if title == "X":
                    pass
                else:
                    self.writer.add_scalar(
                        title, v["data"][step], self.logs["X"]["data"][step]
                    )

    def plot_all(self):
        for step in range(len(self.logs["X"]["data"])):
            for title, v in self.logs.items():
                if v["type"] == "line":
                    if title == "X":
                        pass
                    else:
                        self.writer.add_scalar(
                            title, v["data"][step], self.logs["X"]["data"][step]
                        )

    def plot_line(self, title, vals, X=None):
        if torch.is_tensor(vals):
            vals = vals.detach().cpu().numpy()
        if X is None:
            X = range(len(vals))
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        fig = plt.figure()
        plt.plot(vals)
        self.writer.add_figure(title, fig)

    def plot_bar(self, title, vals, X=None):
        if torch.is_tensor(vals):
            vals = vals.detach().cpu().numpy()
        if X is None:
            X = range(len(vals))
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        fig = plt.figure()
        plt.bar(X, vals)
        self.writer.add_figure(title, fig)

    def plot_bar_stacked(self, title, vals, X=None):
        for i in range(len(vals)):
            if torch.is_tensor(vals[i]):
                vals[i] = vals[i].detach().cpu().numpy()
        if X is None:
            X = range(len(vals[0]))
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        fig = plt.figure()

        plt.bar(X, vals[0])
        sum = vals[0]
        for i in range(1, len(vals)):
            plt.bar(X, vals[i], bottom=sum)
            if torch.is_tensor(vals[i]):
                sum = sum + vals[i]
            else:
                sum = [sum[n] + x for n, x in enumerate(vals[i])]
        self.writer.add_figure(title, fig)

    def plot_heatmap(self, title, vals):
        if torch.is_tensor(vals):
            vals = vals.detach().cpu().numpy()
        fig = plt.figure()
        plt.imshow(vals, cmap="hot", interpolation="nearest")
        self.writer.add_figure(title, fig)

    def print_and_log_losses(
        self,
        stat_train,
        stat_train_memid,
        stat_val,
        stat_val_memid,
        epoch,
        step_lr,
        err=False,
    ):
        self.log("loss::train::{}".format("text"), stat_train.get("loss", 0.0))
        self.log("loss::val::{}".format("text"), stat_val.get("loss", 0.0))
        self.log(
            "text_loss::train::{}".format("text"), stat_train.get("text_loss", 0.0)
        )
        self.log("text_loss::val::{}".format("text"), stat_val.get("text_loss", 0.0))
        self.log("text_err::train::{}".format("text"), stat_train.get("err", 0.0))
        self.log("text_err::val::{}".format("text"), stat_val.get("err", 0.0))
        self.log(
            "ref_obj_loss::train::{}".format("memid"),
            stat_train_memid.get("ref_obj_loss", 0.0),
        )
        self.log(
            "ref_obj_loss::val::{}".format("memid"),
            stat_val_memid.get("ref_obj_loss", 0.0),
        )
        self.log(
            "ref_obj_KL::train::{}".format("memid"),
            stat_train_memid.get("ref_obj_KL", 0.0),
        )
        self.log(
            "ref_obj_KL::val::{}".format("memid"), stat_val_memid.get("ref_obj_KL", 0.0)
        )

        print(
            "\n\t({})\n\ttrain loss: {:.2f}\ttrain ref_obj loss: {:.2f}\ttrain ref_obj KL: {:.2f}\n\tval loss: {:.2f}\t\tval ref_obj loss: {:.2f}\t\tval ref_obj KL: {:.2f}".format(
                "memid",
                stat_train_memid.get("loss", -1.0),
                stat_train_memid.get("ref_obj_loss", -1.0),
                stat_train_memid.get("ref_obj_KL", -1.0),
                stat_val_memid.get("loss", -1.0),
                stat_val_memid.get("ref_obj_loss", -1.0),
                stat_val_memid.get("ref_obj_KL", -1.0),
            )
        )

        print(
            "\n\t({})\n\ttrain loss: {:.2f}\ttrain text loss: {:.2f}\n\tval loss: {:.2f}\tval text loss: {:.2f}".format(
                "text",
                stat_train.get("loss", -1.0),
                stat_train.get("text_loss", -1.0),
                stat_val.get("loss", -1.0),
                stat_val.get("text_loss", -1.0),
            )
        )

        if err:
            print(
                "\ttrain err: {:.2f}%,\tval err: {:.2f}%".format(
                    stat_train.get("err", -1.0) * 100,
                    stat_val.get("err", -1.0) * 100,
                )
            )

        if hasattr(self, "wandb"):
            try:
                self.wandb.log(
                    {
                        "text_loss::train::": stat_train.get("text_loss", 0.0),
                        "text_loss::val::": stat_val.get("text_loss", 0.0),
                        "ref_obj_loss::train::": stat_train_memid.get(
                            "ref_obj_loss", 0.0
                        ),
                        "ref_obj_loss::val::": stat_val_memid.get("ref_obj_loss", 0.0),
                        "ref_obj_KL::train::": stat_train_memid.get("ref_obj_KL", 0.0),
                        "ref_obj_KL::val::": stat_val_memid.get("ref_obj_KL", 0.0),
                    }
                )
            except:
                pass

        self.best_memid_loss = min(
            stat_val_memid.get("loss", float("inf")), self.best_memid_loss
        )
        self.best_memid_ref_obj_loss = min(
            stat_val_memid.get("ref_obj_loss", float("inf")),
            self.best_memid_ref_obj_loss,
        )
        self.best_memid_KL = min(
            stat_val_memid.get("ref_obj_KL", float("inf")), self.best_memid_KL
        )
        self.best_text_loss = min(
            stat_val.get("text_loss", float("inf")), self.best_text_loss
        )
        self.best_text_err = min(stat_val.get("err", float("inf")), self.best_text_err)
        self.best_total_loss = min(
            stat_val.get("loss", float("inf")), self.best_total_loss
        )

        f = open(self.output_file_name, "a")
        f.write("{}".format(epoch))
        f.write(",{:5f}".format(step_lr))
        f.write(",{:3f}".format(stat_train.get("loss", -1.0)))
        f.write(",{:3f}".format(stat_train.get("text_loss", -1.0)))
        f.write(",{:3f}".format(stat_train.get("err", -1.0)))
        f.write(",{:3f}".format(stat_train_memid.get("ref_obj_loss", -1.0)))
        f.write(",{:3f}".format(stat_train_memid.get("ref_obj_KL", -1.0)))
        f.write(",{:3f}".format(stat_val.get("loss", -1.0)))
        f.write(",{:3f}".format(stat_val.get("text_loss", -1.0)))
        f.write(",{:3f}".format(stat_val.get("err", -1.0)))
        f.write(",{:3f}".format(stat_val_memid.get("ref_obj_loss", -1.0)))
        f.write(",{:3f}".format(stat_val_memid.get("ref_obj_KL", -1.0)))
        f.write("\n")
        f.close()

        epoch_loss = [
            epoch,
            step_lr,
            stat_train.get("loss", -1.0),
            stat_train.get("text_loss", -1.0),
            stat_train.get("err", -1.0),
            stat_train_memid.get("ref_obj_loss", -1.0),
            stat_train_memid.get("ref_obj_KL", -1.0),
            stat_val.get("loss", -1.0),
            stat_val.get("text_loss", -1.0),
            stat_val.get("err", -1.0),
            stat_val_memid.get("ref_obj_loss", -1.0),
            stat_val_memid.get("ref_obj_KL", -1.0),
        ]

        self.loss_list.append(epoch_loss)

        print("\n----------------------------------------------------")

        return

    def step(
        self,
        args,
        stat_train,
        stat_val,
        stat_train_memid,
        stat_val_memid,
        elapsed,
        epoch,
        optimizer,
    ):

        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_max_memory_allocated()

        if args.verbosity > 0:
            print(self.log_path)
            step_lr = optimizer.param_groups[0]["lr"]
            print(
                "{}\ts/epoch: {:.1f}\tgpu_mem: {:.1f}gb\tlr: {:.1e}".format(
                    args.ep + 1, elapsed, gpu_mem, step_lr
                )
            )
            if args.cosine_decay or args.exp_decay:
                self.log("compute/lr", step_lr)

            self.print_and_log_losses(
                stat_train,
                stat_train_memid,
                stat_val,
                stat_val_memid,
                epoch,
                step_lr,
                err=True,
            )

            if self.best_memid_loss != float("inf"):
                print("\tbest val memid loss: {:.2f}".format(self.best_memid_loss))
                print(
                    "\tbest val memid ref_obj loss: {:.2f}".format(
                        self.best_memid_ref_obj_loss
                    )
                )
                print(
                    "\tbest val memid KL divergence: {:.2f}".format(self.best_memid_KL)
                )
            if self.best_text_loss != float("inf"):
                print("\tbest val text loss: {:.2f}".format(self.best_text_loss))
            if self.best_text_err != float("inf"):
                print("\tbest val text err: {:.2f}".format(self.best_text_err))
            print("----------------------------------------------------")

        print("\n")

        self.log("X", (args.ep + 1))

        if args.plot:
            self.log("compute/gpu_mem_gb", gpu_mem)
            self.log("compute/epoch_time_s", elapsed)
            self.plot_step(-1)
            self.writer.flush()


def write_granular_losses(granular_losses, clause_type_log, X):
    f = open("granular_losses.csv", "w")

    print(
        "{}, {}, {}, {}, {}".format(
            "key", "total_count", "avg_loss", "total_err/total_count", "avg_err"
        )
    )
    f.write(
        "{}, {}, {}, {}, {}".format(
            "key", "total_count", "avg_loss", "total_err/total_count", "avg_err"
        )
    )
    for key in clause_type_log["loss"].keys():
        avg_loss, avg_err, total_err = 0, 0, 0
        total_count = clause_type_log["count"][key]
        if total_count > 0:
            if key in clause_type_log["loss"]:
                total_loss = clause_type_log["loss"][key]
                avg_loss = total_loss / total_count

            if key in clause_type_log["error"]:
                total_err = clause_type_log["error"][key]
                avg_err = total_err / total_count

            print(
                "{}, {}, {:.2f}, {:.0f}/{}, {:.2f}".format(
                    key, total_count, avg_loss, total_err, total_count, avg_err
                )
            )
            f.write(
                "{}, {}, {:.2f}, {:.0f}/{}, {:.2f}".format(
                    key, total_count, avg_loss, total_err, total_count, avg_err
                )
            )

    print()
    f.write("\n")
    for conj, conj_dict in X.items():
        if conj in ["AND", "OR", "NONE"]:
            if conj_dict["samples"] > 0:
                avg_loss = conj_dict["loss"] / conj_dict["samples"]
                print("{}: {}".format(conj, avg_loss))
                f.write("{}: {}\n".format(conj, avg_loss))

    f.close()
