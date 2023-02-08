from __future__ import print_function
import time
import copy
import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import distributed
from models.detr import build_transformer
from models.gpt import build_gpt
from models.utils import get_vocab_size
from db_dataloader import dataloader
from db_dataloader.dataloader import Collater
from trainer import train
from utils.logger import Logger
import utils.checkpoint as checkpoint
import sys

this_path = os.path.dirname(os.path.abspath(__file__))
data_gen_path = this_path + "/../data/"
sys.path.append(data_gen_path)

from utils.nsm_utils import UUID_HASH_VOCAB_SIZE


def get_parser():
    parser = argparse.ArgumentParser("mycode.train")
    A = parser.add_argument
    # fmt: off
    # model related
    A("--recurrent", action="store_true", default=False, help="use recurrent memory or not")
    A("--nheads", type=int, default=4, help="number of attention heads")
    A("--embedding-dim", type=int, default=128, help="hidden size (embedding dimension), 1 will be added to it for question mark")
    A("--dim-feedforward", type=int, default=512, help="dimension of feed forward net")
    A("--nlabels", type=int, default=148, help="number of output classes")
    A("--enc-layers", type=int, default=2, help="number of encoder layers")
    A("--dropout", type=float, default=0.0, help="dropout rate of ReLU and attention")
    A("--pre-norm", action="store_true", default=False, help="normalize before or not")
    A("--activation", type=str, default="relu", choices=("relu", "gelu", "glu"), help="transformer activation function")
    A("--q_emb", type=str, default="t", choices=("bert", "t", "q"), help="question embedding type")
    A("--model-type", type=str, default="transformer", choices=("transformer", "gpt", "gpt_medium"), help="use pretrained gpt or our transformer")
    A("--pretrained_gpt", action="store_true", default=False, help="")
    A("--memory_encoder", action="store_true", default=False, help="use a memory-only encoder")
    A("--use_lf", action="store_true", default=False, help="")
    A("--context-type", type=str, default="triple_refobj_rel", choices=("triple_refobj_hash", "triple_refobj_rel", "text"), help="triple/ref_obj context or text")
    # optimization related
    A("--lr", type=float, default=0.0001, help="learning rate")
    A("--memid_loss_tradeoff", type=float, default=1.0, help="coefficient of memory match loss")
    A("--memid_loss", action="store_true", default=False, help="only use memory match loss")
    A("--memid_coeff", type=float, default=0.5, help="weight of the memid loss when using both memid and text")
    A("--text_loss", action="store_true", default=False, help="only use text output loss")
    A("--momentum", type=float, default=0.0, help="SGD momentum")
    A("--batch-sz", type=int, default=4, help="batch size")
    A("--test-batch-sz", type=int, default=0, help="set different batch size for test and val data if greater than 0")
    A("--nbatches", type=int, default=-1, help="number of batches in each epoch")
    A("--nepochs", type=int, default=1000, help="number of epochs to train")
    A("--optim", type=str, default="adam", choices=("sgd", "adam"), help="optimization method")
    A("--lr-warmup", type=int, default=100, help="linearly increase LR from 0 during K updates")
    A("--cosine_decay", action="store_true", default=False, help="decay learning rate with cosine scheduler")
    A("--exp_decay", action="store_true", default=False, help="decay learning rate with exponential scheduler")
    A("--attn_supervision", action="store_true", default=False, help="attention softmax loss")
    A("--attn_supervision_type", type=str, default="ce", choices=("ce", "kl", "mask"), help="cross entropy, kl divergence, 1-Y_memid_mask")
    A("--attn_supervision_weight", type=float, default=0.0, help="")
    A("--grad-clip", type=float, default=1, help="clip gradient of each parameter by a given value")
    A("--wdecay", type=float, default=0, help="weight decay")
    A("--update-freq", type=int, default=1, help="(do not use directly, use split-batch instead) update params every K batches. ")
    A("--triple_supervision", action="store_true", default=False, help="use triple supervision in loss")
    A("--large_train", action="store_true", default=False, help="load splits of the dataset")
    A("--text_kl", action="store_true", help="predicts single output vector for text and uses KL to compare against all ground truth tokens")
    A("--accumulation_steps", type=int, default=1, help="if 1, backprop each step")
    # data related
    A("--num_workers", type=int, default=2, help="num of workers in dataloader")
    A("--simple-data-path", type=str, default="")
    A("--episode_dir", type=str, default=f"/private/home/yuxuans/data/neural_memory_data/")
    A("--question_types", nargs="+",default=["color", "width", "height", "count", "spatial", "color_spatial", "spatial_color", "color_spatial_color"])
    A("--num_per_q", type=int, default=1000, help="number of entries per question")
    A("--num_words", type=int, default=4000, help="number of words")
    A("--num_fc_layers", type=int, default=2, help="number of fc layers in featurizer")
    A("--memory_unit", type=str, default="row", help="memory unit",)
    A("--vocab_size", type=int, default=50479, help="50479 is GPT-2 default")
    A("--triples_buffer_size", type=int, default=768, help="triples buffer size (source length)")
    A("--refobjs_buffer_size", type=int, default=32, help="refobjs buffer size (source length)")
    A("--basems_buffer_size", type=int, default=0, help="base mem buffer size (source length)")
    A("--text_buffer_size", type=int, default=768, help="context text buffer size (length)")
    A("--max_seq_len", type=int, default=768, help="")
    A("--max_timesteps", type=int, default=10, help="")
    A("--split_ratio", type=float, default=0.8, help="ratio to split train and val")
    A("--world_sl", type=int, default=60, help="world side length(sl), xyzs should be in the range of [-sl//2, sl//2], used for xyz heads")
    A("--input_type", type=str, default="text", choices=("text", "lf", "both", "random"), help="input data type",)
    A("--tokenizer_path", type=str, default="", help="")
    A("--tokenizer", type=str, default="gpt", choices=("gpt", "simple", "bert"), help="")
    # plotting
    A("--verbosity", type=int, default=1, help="if 0 silent, if 1 plot loss and err, if 2 plot breakdown of errors")
    A("--plot", action="store_true", default=False, help="plot in visdom")
    A("--plot-dir", type=str, default=f"/checkpoint/{os.environ.get('USER')}/tensorboard_runs", help="tensorboard log dir")
    A("--plot-name", type=str, default="", help="tensorboard log name (default: datetime)")
    A("--plot-checkpoint", action="store_true", default=False, help="just re-plot all steps from a checkpoint and exit")
    A("--wandb", action="store_true", default=False, help="use wandb")
    A("--wandb_entity", type=str, default="", help="wandb login username")
    A("--wandb_dir", type=str, default="", help="wandb dir")
    A("--wandb_project", type=str, default="", help="wandb proj name")
    # misc
    A("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    A("--multi_gpu", action="store_true", default=False, help="nn.DataParallel the model")
    A("--checkpoint_dir", type=str, default="", help="dir to save/load model")
    A("--checkpoint-freq", type=int, default=0, help="how often to keep a copy")
    A("--load-only", action="store_true", default=False, help="do not save to checkpoint")
    A("--load-checkpoint", type=str, default=None, help="path to checkpoint to load and finetune from")
    A("--full-test", action="store_true", default=False, help="do testing on whole data")
    A("--normalize_floats", action="store_true", default=False, help="normalize float vals in featurizer to [0,1] range")
    A("--fp16", action="store_true", default=False, help="fp16 training with Apex")
    A("--lazy-load-data", action="store_true", default=False, help="moves data to GPU one sample at a time")
    A("--seed", type=int, default=52, help="random seed")
    A("--run_name", type=str, default="", help="name for this particular run")
    A("--overwrite-logs", action="store_true", default=False, help="overwrite logs")
    A("--overwrite-checkpoint", action="store_true", default=False, help="overwrite model checkpoint")
    A("--no_checkpoint", action="store_true", default=False, help="don't save model")
    A("--debug", action="store_true", default=False, help="don't log or save model")
    A("--interact", action="store_true", default=False, help="")
    A("--disable_tqdm", action="store_true", default=False, help="")
    A("--print_err", action="store_true", default=False, help="only print incorrect preds in interact mode")
    A("--text_out", action="store_true", default=False, help="generate a text output")
    
    distributed.add_args(parser)
    # fmt: on
    return parser


def update_args(args):
    if args.head_dim == 0:
        assert args.hid_sz % args.nheads == 0
        args.head_dim = args.hid_sz // args.nheads

    if args.split_batch > 1:
        assert args.batch_sz % args.split_batch == 0
        assert args.test_batch_sz % args.split_batch == 0
        args.update_freq = args.split_batch
        args.batch_sz = args.batch_sz // args.split_batch
        args.test_batch_sz = args.test_batch_sz // args.split_batch
        args.nbatches *= args.split_batch
        args.lr_warmup *= args.split_batch

    if args.plot and args.plot_name == "":
        args.plot_name = time.strftime("%Y%m%d_%H%M%S")

    if args.test_batch_sz == 0:
        args.test_batch_sz = args.batch_sz


def main(args, logger=None):
    # print environment
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")

    args = copy.deepcopy(args)  # important for requeue!!! don't change original args

    args.uuid_hash_vocab_size = UUID_HASH_VOCAB_SIZE
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.interact:
        args.disable_tqdm = True

    batch_sz = args.batch_sz  # store initial batch_sz before distributed splits it up

    if logger is None:
        logger = Logger(args, batch_sz)

    args.checkpoint = os.path.join(logger.log_path, "checkpoint_best_val_text_err.pt")

    # deterministic
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.distributed:
        distributed.init(args)
    args.device = torch.device("cuda" if use_cuda else "cpu")

    # load data
    data_dir = args.simple_data_path
    args.nlabels = 1
    tdict = None

    train_sampler, val_sampler = None, None  # can replace with custom sampler
    train_dataset = None

    if args.full_test:
        val_dataset = dataloader.get_presplit_data(args, data_dir)
        train_dataset = val_dataset
    elif args.large_train:
        train_file = dataloader.get_training_chunk(data_dir)
        print("loading {}".format(train_file))
        train_dataset = dataloader.get_presplit_data(args, train_file)
        val_dataset = dataloader.get_presplit_data(
            args, os.path.join(data_dir, "val.pth")
        )
    else:
        # this will split the training data into a train/val split
        train_dataset, val_dataset = dataloader.get_data(
            args, os.path.join(data_dir, "train_1_1.pth")
        )

    dataloader_train, sampler_train = distributed.wrap_dataset(
        args, train_dataset, Collater(args, train=True), sampler=train_sampler
    )
    dataloader_val, sampler_val = distributed.wrap_dataset(
        args, val_dataset, Collater(args, train=False), sampler=val_sampler, test=True
    )

    if args.nbatches == -1:
        args.nbatches = int(len(train_dataset) / args.batch_sz)

    # create a model
    if args.model_type in ["gpt", "gpt_medium"]:
        model = build_gpt(args)
    else:
        vocab_size = get_vocab_size(args)
        model = build_transformer(args, vocab_size)

    if args.multi_gpu and torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model.to(args.device)

    # count params
    nparameters = 0
    params = []
    for param in model.parameters():
        if param.requires_grad:
            nparameters += param.numel()
            params.append(param)
    logger.print("nparameters={:.2f}M".format(nparameters / 1e6))

    print("nparameters={:.2f}M".format(nparameters / 1e6))

    # OPTIM param
    if args.optim == "sgd":
        optimizer = optim.SGD(
            params, lr=args.lr, weight_decay=args.wdecay, momentum=args.momentum
        )
    elif args.optim == "adam":
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

    if args.cosine_decay:
        # will do warm-up manually later
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.nepochs * args.nbatches
        )
    elif args.exp_decay:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    elif args.lr_warmup > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: min(1, ep / args.lr_warmup)
        )
    else:
        scheduler = None

    if args.fp16:
        from apex import amp

        model, optimizer = amp.initialize(model, optimizer)

    model = distributed.wrap_model(args, model)

    ep_init = checkpoint.load(args, model, optimizer, logger, scheduler)

    if args.plot_checkpoint:
        logger.plot_all()
        return

    if args.full_test:
        # perform evaluation only
        with torch.no_grad():
            stat_val, stat_val_memid = train(
                args,
                model,
                optimizer,
                scheduler,
                dataloader_val,
                epoch=0,
                test_only=True,
            )
            if args.distributed:
                distributed.collect_stat(
                    args, stat_val, stat_val_memid, stat_val, stat_val_memid
                )

            if not args.distributed or args.rank == 0:
                print(
                    "val text loss: {:.3f}\tval ref_obj kl loss: {:.3f}".format(
                        stat_val.get("text_loss", -1.0),
                        stat_val_memid.get("ref_obj_KL", -1.0),
                    )
                )
                print("val err: {:.3f}".format(stat_val.get("err", -1.0)))
        return

    for ep in range(ep_init, args.nepochs):
        if args.large_train:
            train_file = dataloader.get_training_chunk(data_dir)
            print("loading {}".format(train_file))
            train_dataset = dataloader.get_presplit_data(args, train_file)
            dataloader_train, sampler_train = distributed.wrap_dataset(
                args, train_dataset, Collater(args, train=True), sampler=None
            )

        if args.distributed:
            sampler_train.set_epoch(ep)
            sampler_val.set_epoch(ep)

        t_sta = time.time()
        args.ep = ep
        stat_train, stat_train_memid = train(
            args, model, optimizer, scheduler, dataloader_train, epoch=ep
        )

        elapsed = time.time() - t_sta
        with torch.no_grad():
            stat_val, stat_val_memid = train(
                args,
                model,
                optimizer,
                scheduler,
                dataloader_val,
                epoch=ep,
                test_only=True,
            )

        if args.distributed:
            distributed.collect_stat(
                args, stat_train, stat_train_memid, stat_val, stat_val_memid
            )

        if args.distributed == False or args.rank == 0:
            # only the master process will do logging, plotting and checkpoint
            logger.step(
                args,
                stat_train,
                stat_val,
                stat_train_memid,
                stat_val_memid,
                elapsed,
                ep,
                optimizer,
            )
            checkpoint.save(
                args,
                model,
                optimizer,
                logger,
                scheduler,
                tdict,
                stat_val,
                stat_val_memid,
            )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = Logger(args, args.batch_sz)
    logger.print(str(args))

    try:
        main(args, logger)
    except KeyboardInterrupt:
        if hasattr(logger, "wandb"):
            logger.wandb.finish()
        exit()

    if hasattr(logger, "wandb"):
        logger.wandb.finish()
