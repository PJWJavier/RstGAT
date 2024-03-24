import torch
import numpy as np
from math import ceil
def merge_str(a,b):
    return str(a)+str(b)

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

def data_padding(max_len,seq_tensor):
    x = torch.tensor([0]*max_len)
    trunc = seq_tensor[:max_len]
    x[:len(trunc)] = trunc
    return x

def Bart_data_padding(max_len,seq_tensor,pad = 1):
    x = torch.tensor([pad]*max_len)
    trunc = seq_tensor[:max_len]
    x[:len(trunc)] = trunc
    return x

def para_data_padding(max_len,max_sent_len, para_tensor):
    x = torch.zeros(max_sent_len, max_len)
    trunc = para_tensor[:max_sent_len]
    x[:len(trunc)] = trunc
    return x

def _get_total_training_steps(args, Dataset) -> int:
    return ceil(len(Dataset) / args.batch_size) * args.num_epoch

# Model Test Mode
def print_args(model):
    n_trainable_params, n_nontrainable_params = 0, 0
    for p in model.parameters():
        n_params = torch.prod(torch.tensor(p.shape))
        if p.requires_grad:
            n_trainable_params += n_params
        else:
            n_nontrainable_params += n_params
    print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
    print('> training arguments:')

def TScollate(batch):
    rst, polarity,map , sent_num, ref_num, input_tensor, edu_input_tensor, reference_tensor, ref_segment_tensor, segment_tensor = None, None, None, None, None, None, None, None, None, None
    for item in batch:
        if rst == None:
            rst, polarity, map, sent_num, ref_num, input_tensor, edu_input_tensor, reference_tensor, ref_segment_tensor, segment_tensor = [item["rst"]], \
            [item["polarity"]], [item["map"]], [item["sent_num"]], [item["ref_num"]], item["input_tensor"], item["edu_input_tensor"], item["reference_tensor"], \
            item["ref_segment_tensor"], item["segment_tensor"]
        else:
            rst.append(item["rst"])
            polarity.append(item["polarity"])
            map.append(item["map"])
            sent_num.append(item["sent_num"])
            ref_num.append(item["ref_num"])
            input_tensor = torch.stack([input_tensor,item["input_tensor"]],dim=0)
            edu_input_tensor = torch.stack([edu_input_tensor, item["edu_input_tensor"]], dim=0)
            reference_tensor = torch.stack([reference_tensor, item["reference_tensor"]], dim=0)
            ref_segment_tensor = torch.stack([ref_segment_tensor, item["ref_segment_tensor"]], dim=0)
            segment_tensor = torch.stack([segment_tensor, item["segment_tensor"]], dim=0)
    ret_dict = {"rst":rst, "polarity":polarity, "map":map, "sent_num":sent_num, "ref_num":ref_num, "input_tensor":input_tensor
        , "edu_input_tensor":edu_input_tensor, "reference_tensor":reference_tensor, "ref_segment_tensor":ref_segment_tensor, "segment_tensor":segment_tensor}
    return ret_dict