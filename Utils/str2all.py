from Tokenizer.ModelTokenizer import BertTokenizer, CxGBertTokenizer
from DataProcessor.Dataset.BertDataset import BertDataset
# from Model.Components.embedding import BertEmbedding
# from Model.Core.BertModel import BertTarget
# from Model.Encoder.TransformerEncoder import TransformerEncoder
from DataProcessor.DataLoader import BertDataLoader
from Model.Trainer import BertTrainer, BaseTrainer
# from Utils.act_fun import *
from Utils.optimizers import *

str2tokenizer  = {'bert' : BertTokenizer, 'cxgbert' : CxGBertTokenizer}
str2dataset    = {'bert' : BertDataset}

# str2act        = {"gelu": gelu, "gelu_fast": gelu_fast, "relu": relu, "linear": linear}
# str2embedding  = {"bert": BertEmbedding}
# str2target     = {"bert": BertTarget}
# str2encoder    = {"transformer": TransformerEncoder}

str2dataloader = {"bert": BertDataLoader}
str2trainer    = {"bert": BertTrainer}
str2optimizer  = {"adamw": AdamW, "adafactor": Adafactor}

str2scheduler  = {"linear": get_linear_schedule_with_warmup, "cosine": get_cosine_schedule_with_warmup,
                 "cosine_with_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
                 "polynomial": get_polynomial_decay_schedule_with_warmup,
                 "constant": get_constant_schedule, "constant_with_warmup": get_constant_schedule_with_warmup}