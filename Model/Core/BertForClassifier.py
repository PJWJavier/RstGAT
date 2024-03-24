from Model.model import load_model
import torch
import torch.nn as nn
from Utils import str2embedding, str2encoder
import os

class BaseModelOutputForClassifier(nn.Module):
    def __init__(self, last_hidden_state : torch.Tensor = None, pooler_output : torch.Tensor = None, attentions : torch.Tensor = None):
        super(BaseModelOutputForClassifier, self).__init__()
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output
        self.attentions = attentions

    def __getitem__(self, item):
        if item in self.__dict__.keys():
            return self.__dict__[item]
        else:
            raise Exception('BaseModelOutputForClassifier can not get attribute {}, please check.'.format(item))

    def __len__(self):
        var_num = 0
        for var in self.__dict__.keys():
            if self.__dict__[var] is not None: var_num += 1
        return var_num

class BertForClassifierModel(nn.Module):
    def __init__(self, args):
        super(BertForClassifierModel, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder   = str2encoder[args.encoder](args)
        self.pooling   = args.pooling
        self.fc        = nn.Linear(args.hidden_size, 1)

    def from_pretrained(self, model_path):
        if os.path.exists(model_path) is not True:
            raise Exception('BertForClassifierModel can not get param files, at {}. Please check.'.format(model_path))
        else:
            load_model(self, model_path)

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        last_hidden_state = self.encoder(emb, seg)
        # if self.pooling   == "mean":  output = torch.mean(last_hidden_state, dim=1)
        # elif self.pooling == "max": output = torch.max(last_hidden_state, dim=1)[0]
        # elif self.pooling == "last":output = last_hidden_state[:, -1, :]
        # else: output      = last_hidden_state[:, 0, :]
        output            = torch.tanh(self.fc(last_hidden_state))
        return BaseModelOutputForClassifier(last_hidden_state=last_hidden_state, pooler_output=output)