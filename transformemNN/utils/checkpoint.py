import os
import shutil
import torch
import pickle


def load_path(args, path):
    print("loading from " + path)
    if args.distributed:
        # the model is saved from gpu0 so we need to map it to CPU first
        return torch.load(path, map_location=lambda storage, loc: storage)
    else:
        return torch.load(path)


def load_checkpoint(args, path, model, optimizer, logger, scheduler):
    f = load_path(args, path)
    ep_init = f["epoch"]
    model.load_state_dict(f["model"])
    logger.set_state(f["logger"])
    optimizer.load_state_dict(f["optimizer"])
    logger.loss_list = f["losses"]
    if "scheduler_epoch" in f:
        # scheduler.load_state_dict(f['scheduler'])
        scheduler.step(f["scheduler_epoch"])
    if "amp" in f:
        from apex import amp

        amp.load_state_dict(f["amp"])

    print("loaded checkpoint from {}".format(path))
    return ep_init


def load_checkpoint(args, source_path, model):
    f = load_path(args, source_path)
    model.load_state_dict(f["model"], strict=False)
    # Start from the beginning
    return 0


def load(args, model, optimizer, logger, scheduler):
    ep_init = 0
    if os.path.exists(args.checkpoint) and not args.overwrite_checkpoint:
        try:
            ep_init = load_checkpoint(
                args, args.checkpoint, model, optimizer, logger, scheduler
            )
        except Exception as e:
            print(f"load failed: {e}")
            # try the backup checkpoint
            if os.path.exists(args.checkpoint + ".bak"):
                try:
                    ep_init = load_checkpoint(
                        args,
                        args.checkpoint + ".bak",
                        model,
                        optimizer,
                        logger,
                        scheduler,
                    )
                except Exception as e:
                    print(f"backup load failed: {e}")
    elif args.load_checkpoint and os.path.exists(args.load_checkpoint):
        ep_init = load_checkpoint(args, source_path=args.load_checkpoint, model=model)

    return ep_init


def save(args, model, optimizer, logger, scheduler, tdict, stat_val, stat_val_memid):
    f = dict()
    f["epoch"] = args.ep + 1
    f["model"] = model.state_dict()
    f["logger"] = logger.get_state()
    f["optimizer"] = optimizer.state_dict()
    f["tdict"] = tdict
    f["losses"] = logger.loss_list

    if args.fp16:
        from apex import amp

        f["amp"] = amp.state_dict()

    if scheduler is not None:
        f["scheduler_epoch"] = scheduler.last_epoch

    if args.checkpoint != "" and args.load_only == False:
        if os.path.exists(args.checkpoint):
            if (
                args.checkpoint_freq > 0
                and args.ep > 0
                and args.ep % args.checkpoint_freq == 0
            ):
                try:
                    shutil.copyfile(
                        args.checkpoint, args.checkpoint + "." + str(args.ep)
                    )
                except:
                    print("save copy failed")
            # make a backup in case this save fails
            try:
                os.replace(args.checkpoint, args.checkpoint + ".bak")
            except:
                print("save backup failed")

        try:
            torch.save(f, args.checkpoint)
        except:
            print("checkpoint save failed")

        # try:
        #     if hasattr(logger, 'wandb'):
        #         logger.wandb.save(args.checkpoint)
        # except:
        #     print('wandb save failed')

    if logger.log_path != "":
        text_loss = stat_val.get("text_loss", None)
        text_err = stat_val.get("err", None)
        mem_id_loss = stat_val_memid.get("ref_obj_loss", None)
        total_loss = stat_val_memid.get("loss", None)

        if (
            total_loss
            and (total_loss <= logger.best_total_loss)
            and (not args.no_checkpoint)
        ):
            print(
                "saving checkpoint to {}".format(
                    os.path.join(logger.log_path, "checkpoint_best_val.pt")
                )
            )
            torch.save(f, os.path.join(logger.log_path, "checkpoint_best_val.pt"))

        if (
            text_loss
            and (text_loss <= logger.best_text_loss)
            and (not args.no_checkpoint)
        ):
            print(
                "saving checkpoint to {}".format(
                    os.path.join(logger.log_path, "checkpoint_best_val_text.pt")
                )
            )
            torch.save(f, os.path.join(logger.log_path, "checkpoint_best_val_text.pt"))

        if text_err and (text_err <= logger.best_text_err) and (not args.no_checkpoint):
            print(
                "saving checkpoint to {}".format(
                    os.path.join(logger.log_path, "checkpoint_best_val_text_err.pt")
                )
            )
            torch.save(
                f, os.path.join(logger.log_path, "checkpoint_best_val_text_err.pt")
            )

        if (
            mem_id_loss
            and (mem_id_loss <= logger.best_memid_ref_obj_loss)
            and (not args.no_checkpoint)
        ):
            print(
                "saving checkpoint to {}".format(
                    os.path.join(logger.log_path, "checkpoint_best_val_memid.pt")
                )
            )
            torch.save(f, os.path.join(logger.log_path, "checkpoint_best_val_memid.pt"))

        print("saving losses to {}".format(os.path.join(logger.log_path, "losses.pkl")))
        with open(os.path.join(logger.log_path, "losses.pkl"), "wb") as fp:
            pickle.dump(logger.loss_list, fp)
