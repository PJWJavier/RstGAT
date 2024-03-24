import os
import numpy as np
import random
import torch

def count_lines(file_path):
    lines_num = 0
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(2**20)
            if not data:
                break
            lines_num += data.count(b'\n')
    return lines_num

def set_seed(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
