import torch
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,degree
import torch.nn as nn
from Model.Components.attention import AdditiveAttention

class GAT(nn.Module):
    def __init__(self,seq_len,featrues_dim, device):
        super(GAT,self).__init__()
        self.atten = AdditiveAttention(seq_len,seq_len,seq_len,0.1)
        self.fc = nn.Linear(seq_len * 2, seq_len)
    def forward(self,inputs,input_masks,edge_index):
        edge_len = edge_index.shape[1]
        valid_lens = torch.sum(input_masks,dim=1).unsqueeze(1)
        for i in range(edge_len):
            x = int(edge_index[0][i])
            emb_x = inputs[x]
            valid_x = valid_lens[x]
            y = int(edge_index[1][i])
            emb_y = inputs[y]
            valid_y = valid_lens[y]
            y2x = self.atten(emb_x,emb_y,emb_y,valid_x)
            x2y = self.atten(emb_y, emb_x, emb_x, valid_y)
            x_out = self.fc(torch.cat((emb_x,y2x),dim=0).T).T
            y_out = self.fc(torch.cat((emb_y, x2y), dim=0).T).T
            inputs[edge_index[0][i]] = x_out
            inputs[edge_index[1][i]] = y_out
        return inputs


