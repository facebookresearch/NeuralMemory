import torch
import torch.nn as nn
import torch.utils.data as tds
import submitit


def add_args(parser):
    parser.add_argument(
        "--distributed", action="store_true", default=False, help="distributed training"
    )
    parser.add_argument("--local_rank", type=int, default=0, help="")
    parser.add_argument(
        "--submitit", action="store_true", default=False, help="using submitit"
    )
    parser.add_argument(
        "--dist-init", type=str, default="", help="distributed training"
    )
    parser.add_argument(
        "--parallel", action="store_true", default=False, help="run parallel training"
    )
    parser.add_argument("--world_size", type=int, default=2, help="")


def init(args):
    if args.submitit:
        job_env = submitit.JobEnvironment()
        args.local_rank = job_env.local_rank
        args.rank = job_env.global_rank
        args.world_size = job_env.num_tasks
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.dist_init,
            rank=job_env.global_rank,
            world_size=job_env.num_tasks,
        )
    else:
        torch.distributed.init_process_group(backend="nccl", init_method='file:///tmp/transformemNN_dist_init', world_size=args.world_size, rank=args.local_rank)
        # torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=- 1, rank=- 1)
        args.rank = torch.distributed.get_rank()
        # args.world_size = torch.distributed.get_world_size()
    args.batch_sz = args.batch_sz // args.world_size
    torch.cuda.set_device(args.local_rank)
    if args.rank > 0:
        args.plot = False
        # args.wandb = False


def split_data(args, train_data, val_data, test_data):
    assert args.batch_sz % args.world_size == 0
    args.batch_sz = args.batch_sz // args.world_size
    train_data = train_data[args.batch_sz * args.rank : args.batch_sz * (args.rank + 1)]
    if args.test_batch_sz < args.world_size:
        # sometimes small test batch size is needed
        r = args.rank % args.test_batch_sz
        val_data = val_data[r : r + 1]
        test_data = test_data[r : r + 1]
        args.test_batch_sz = 1
    else:
        assert args.test_batch_sz % args.world_size == 0
        args.test_batch_sz = args.test_batch_sz // args.world_size
        val_data = val_data[
            args.test_batch_sz * args.rank : args.test_batch_sz * (args.rank + 1)
        ]
        test_data = test_data[
            args.test_batch_sz * args.rank : args.test_batch_sz * (args.rank + 1)
        ]
    return train_data, val_data, test_data


def wrap_model(args, model):
    if args.distributed:
        model = model.to(args.device)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    elif args.parallel:
        model = torch.nn.DataParallel(model)
        model = model.to(args.device)
    else:

        class DummyWrapper(nn.Module):
            def __init__(self, mod):
                super(DummyWrapper, self).__init__()
                self.module = mod

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        model = DummyWrapper(model)
        model = model.to(args.device)
    return model

def wrap_dataset(args, dataset, collater, sampler=None, test=False):
    if args.submitit:
        job_env = submitit.JobEnvironment()
        sampler = tds.distributed.DistributedSampler(
            dataset, num_replicas=job_env.num_tasks, rank=job_env.global_rank
        )
    elif args.distributed:
        sampler = tds.distributed.DistributedSampler(dataset)
    batch_sz = args.batch_sz
    if test:
        shuffle=False
    else:
        shuffle = (sampler is None)

    if args.full_test:
        drop_last = False
    else:
        drop_last = True
        
    dataloader = tds.DataLoader(
        dataset, collate_fn=collater, batch_size=batch_sz, shuffle=shuffle, sampler=sampler, drop_last=drop_last, num_workers=args.num_workers
    )
    return dataloader, sampler


def collect_stat(args, stat_train, stat_train_memid, stat_val, stat_val_memid):
    X = torch.zeros(8).to(args.device)
    X[0] = stat_train["loss"]
    X[1] = stat_train.get("text_loss", -1.0)
    X[2] = stat_train_memid.get("ref_obj_loss", -1.0)
    X[3] = stat_train_memid.get("ref_obj_KL", -1.0)

    X[4] = stat_val["loss"]
    X[5] = stat_val.get("text_loss", -1.0)
    X[6] = stat_val_memid.get("ref_obj_loss", -1.0)
    X[7] = stat_val_memid.get("ref_obj_KL", -1.0)

    torch.distributed.reduce(X, 0)
    torch.cuda.synchronize()
    
    if args.rank == 0:
        stat_train["loss"] = X[0].item() / args.world_size
        stat_train["text_loss"] = X[1].item() / args.world_size
        stat_train_memid["ref_obj_loss"] = X[2].item() / args.world_size
        stat_train_memid["ref_obj_KL"] = X[3].item() / args.world_size

        stat_val["loss"] = X[4].item() / args.world_size
        stat_val["text_loss"] = X[5].item() / args.world_size
        stat_val_memid["ref_obj_loss"] = X[6].item() / args.world_size
        stat_val_memid["ref_obj_KL"] = X[7].item() / args.world_size

