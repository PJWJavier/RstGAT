# Import Base libs
import sys
import os
parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)
from math import ceil
from Model import ADFA_Trainer
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BartTokenizer, BartForConditionalGeneration,AutoTokenizer
from dataset.WikiTSDaraset import WikiTSDaraset
from dataset.efactdataset import efactdataset
from Model.RSTBART_AS import RSTBART_AS
from Model.BART_RST import BART_RST
from configs import GLUE_opts
from Utils.str2all import str2tokenizer
from Utils.hyperparam import load_hyperparam
from Model.Utils import _get_total_training_steps, print_args
from Model.Summary_Trainer import Trainer

import torch.multiprocessing as mp

# from Model.Bart_base_Trainer import Trainer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# from apex import amp

import copy
# Model Train Mode


def ADFAcollate(batch):
    plain_article, rst_graph, polarity,map , sent_num, ref_num, input_tensor, reference_tensor, ref_segment_tensor, segment_tensor = None, None, None, None, None, None, None, None, None, None
    for item in batch:
        if rst_graph == None:
            plain_article, rst_graph, sent_num, input_tensor, reference_tensor, ref_segment_tensor, segment_tensor = [item["plain_article"]], torch.unsqueeze(item["rst"],0), \
              torch.unsqueeze(item["sent_num"],0), torch.unsqueeze(item["input_tensor"],0), torch.unsqueeze(item["reference_tensor"],0), \
            torch.unsqueeze(item["ref_segment_tensor"],0), torch.unsqueeze(item["segment_tensor"],0)
        else:
            plain_article.append(item["plain_article"])
            rst_graph = torch.concat([rst_graph , torch.unsqueeze(item["rst"],0)],dim=0)
            sent_num = torch.concat([sent_num,torch.unsqueeze(item["sent_num"],0)],dim=0)
            input_tensor = torch.concat([input_tensor,torch.unsqueeze(item["input_tensor"],0)],dim=0)
            reference_tensor = torch.concat([reference_tensor, torch.unsqueeze(item["reference_tensor"],0)], dim=0)
            ref_segment_tensor = torch.concat([ref_segment_tensor, torch.unsqueeze(item["ref_segment_tensor"],0)], dim=0)
            segment_tensor = torch.concat([segment_tensor, torch.unsqueeze(item["segment_tensor"],0)], dim=0)
    ret_dict = {"plain_article":plain_article,"rst":rst_graph, "sent_num":sent_num,"input_tensor":input_tensor
        , "reference_tensor":reference_tensor, "ref_segment_tensor":ref_segment_tensor, "segment_tensor":segment_tensor}
    return ret_dict

# def ADFAcollate(batch):
#     batch_new = copy.deepcopy(batch)
#     rst = []
#     polarity = []
#     map = []
#     for item in batch_new:
#
#         rst.append(item["rst"])
#         item.pop("rst")
#         polarity.append(item["polarity"])
#         item.pop("polarity")
#         map.append(item["map"])
#         item.pop("map")
#
#     return torch.utils.data.dataloader.default_collate(batch_new), rst, polarity, map

def run(gpu,args):
    # Device
    rank = args.nr * args.gpus + gpu
    device = "cuda:" + str(gpu)
    device      = torch.device(device)
    args.device = device
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    # Visualizer
    # Visual      = VIS(args)
    # Dataset & DataLoader

    Trainset = efactdataset(args, args.trainset_path, args.max_seq_len)
    Evalset = efactdataset(args, args.testset_path, args.max_seq_len)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        Trainset,
        num_replicas=args.world_size,
        rank=rank
    )
    eval_sampler = torch.utils.data.distributed.DistributedSampler(
        Evalset,
        num_replicas=args.world_size,
        rank=rank
    )
    TrainLoader = DataLoader(dataset=Trainset, batch_size=args.batch_size, shuffle=False,num_workers=0,
       pin_memory=True,sampler=train_sampler,collate_fn=ADFAcollate)
    EvalLoader = DataLoader(dataset=Evalset, batch_size=args.batch_size, shuffle=False,num_workers=0,
       pin_memory=True,sampler=eval_sampler,collate_fn=ADFAcollate)

    # Model & Trainer
    #就这一行
    # model = RSTBART_AS(args).to(args.device)

    model = RSTBART_AS(args).to(args.device)
    _params = model.parameters()
    optimizer = args.optimizer(_params, lr=args.learning_rate, weight_decay=args.l2reg, correct_bias=True)
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    criterion = nn.CrossEntropyLoss()


    num_warmup_steps = 100
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, _get_total_training_steps(args, Trainset))
    if args.continue_training:
        checkpoint = torch.load('state_dict/pre_0.9816_checkpoint.pth',map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        print("checkpoint loaded")

    print_args(model)
    if args.device.type == 'cuda':
        print("cuda memory allocated:", torch.cuda.memory_allocated(device=args.device.index))
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O2")


    trainer     = Trainer(args, model, criterion, optimizer, scheduler, device, rank)
    # trainer.eval(EvalLoader)
    # Training
    max_test_precision, max_f1, max_recall = trainer.train(TrainLoader, EvalLoader)
    with open("GLUE_Record.txt",'a') as f:
        f.write(str(round(max_test_precision, 4)) + " " + str(round(max_f1, 4))+ " " + str(round(max_recall, 4)) + '\n')
    f.close()



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    GLUE_opts(parser)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 3"
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.polarity_dim = 5
    if args.optimizer == "bertAdam":
        args.optimizer = AdamW
    args.inputs_cols = ['all_indices_tensor', 'segment_tensor', 'polarity']
    args.trainset_path = 'Data/efact/' + 'train_rst_end.csv'
    args.testset_path = 'Data/efact/' + 'test_rst_end.csv'
    args.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn', cache_dir='./BART_large',
                                                   local_files_only=True)
    args.vocab_size = args.tokenizer.vocab_size
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device is None else torch.device(args.device)

    args.continue_training = True
    args.max_sent_len = 40
    if args.config_path:
        args = load_hyperparam(args)

    args.world_size = args.gpus * args.nodes  #
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10026'
    mp.spawn(run, nprocs=args.gpus, args=(args,))



if __name__ == '__main__':
    main()
