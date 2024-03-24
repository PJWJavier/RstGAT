from Utils.misc import set_seed
from Utils.str2all import str2tokenizer, str2optimizer, str2scheduler, str2dataloader, str2trainer
from Model.model import build_model, load_model
from Model.Core.BertForClassifier import BertForClassifierModel
import math
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

def train_and_validate(args):
    set_seed(args.seed)

    # Load vocabulary.
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    args.vocab = args.tokenizer.vocab

    # Build model.
    model = BertForClassifierModel(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.from_pretrained(args.pretrained_model_path)
    else:
        # Initialize with normal distribution.
        if args.deep_init:
            scaled_factor = 1 / math.sqrt(2.0 * args.layers_num)
            for n, p in list(model.named_parameters()):
                if "gamma" not in n and "beta" not in n:
                    if "linear_2.weight" in n or "final_linear.weight" in n:
                        p.data.normal_(0, 0.02 * scaled_factor)
                    elif "linear_2.bias" in n or "final_linear.bias" in n:
                        p.data.zero_()
                    else:
                        p.data.normal_(0, 0.02)
        else:
            for n, p in list(model.named_parameters()):
                if "gamma" not in n and "beta" not in n:
                    p.data.normal_(0, 0.02)

    if args.deepspeed:
        worker(args.local_rank, None, args, model)
    elif args.dist_train:
        # Multiprocessing distributed mode.
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model), daemon=False)
    elif args.single_gpu:
        # Single GPU mode.
        worker(args.gpu_id, None, args, model)
    else:
        # CPU mode.
        worker(None, None, args, model)

def worker(proc_id, gpu_ranks, args, model):
    """
    Args:
        proc_id: The id of GPU for single GPU mode;
                 The id of process (and GPU) for multiprocessing distributed mode.
        gpu_ranks: List of ranks of each process.
    """
    set_seed(args.seed)

    if args.deepspeed:
        import deepspeed
        deepspeed.init_distributed(dist_backend=args.backend)
        rank = dist.get_rank()
        gpu_id = proc_id
    elif args.dist_train:
        rank = gpu_ranks[proc_id]
        gpu_id = proc_id
    elif args.single_gpu:
        rank = None
        gpu_id = proc_id
    else:
        rank = None
        gpu_id = None

    if args.dist_train:
        train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, rank, args.world_size, True)
    else:
        train_loader = str2dataloader[args.target](args, args.dataset_path, args.batch_size, 0, 1, True)

    # Build optimizer.
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    if args.optimizer in ["adamw"]:
        custom_optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        custom_optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer, args.total_steps*args.warmup)
    else:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer, args.total_steps*args.warmup, args.total_steps)

    if args.deepspeed:
        optimizer = None
        scheduler = None

        # IF User NOT defined optimezer in deepspeed config,
        # Then use Self Defined Optimizer
        if "optimizer" not in args.deepspeed_config_param:
            optimizer = custom_optimizer
            if args.local_rank == 0:
                print("Use Custum Optimizer", optimizer)
        if "scheduler" not in args.deepspeed_config_param:
            scheduler = custom_scheduler
            if args.local_rank == 0:
                print("Use Custom LR Schedule", scheduler)
        model, optimizer, _, scheduler = deepspeed.initialize(
                                                    model=model,
                                                    model_parameters=optimizer_grouped_parameters,
                                                    args=args,
                                                    optimizer=optimizer,
                                                    lr_scheduler=scheduler,
                                                    mpu=None,
                                                    dist_init_required=False)
    else:
        if gpu_id is not None:
            model.cuda(gpu_id)
        optimizer = custom_optimizer
        scheduler = custom_scheduler
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
            args.amp = amp

        if args.dist_train:
            # Initialize multiprocessing distributed training environment.
            dist.init_process_group(backend=args.backend,
                                    init_method=args.master_ip,
                                    world_size=args.world_size,
                                    rank=rank)
            model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
            print("Worker %d is training ... " % rank)
        else:
            print("Worker is training ...")

    trainer = str2trainer[args.target](args)
    trainer.train(args, gpu_id, rank, train_loader, model, optimizer, scheduler)
