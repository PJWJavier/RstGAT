import torch
import torch.nn as nn
import torch_scatter
from Model.Core.BertForClassifier import BertForClassifierModel
from Model.Components.GAT import GATConv
from Model.Components.decoder import DecoderRNN
def data_padding(max_len,seq_tensor):
    x = torch.tensor([0.0]*max_len).cuda(seq_tensor.device)
    trunc = seq_tensor[:max_len]
    x[:trunc.shape[0]] = trunc
    return x

class BERT_TS(nn.Module):
    def __init__(self, args):
        super(BERT_TS, self).__init__()
        self.args = args
        self.bert = BertForClassifierModel(args)
        self.GAT = GATConv(args.bert_dim, args.hidden_size, heads=1)
        self.bert.from_pretrained(args.pretrained_model)
        self.max_sent_length = 128
        self.sc = nn.Linear(args.hidden_size, args.polarity_dim)
        self.decoder = nn.TransformerDecoderLayer(args.hidden_size, nhead=1)
        self.fc_out = nn.Linear(args.hidden_size, 30522)
    def forward(self, src, seg,reference_tensor,ref_segment_tensor, sent_num, ref_num, rst, polarity, map):
        res = []
        for i in range(src.shape[0]):
            res.append(self.bert(src[i], seg[i]))
        ref = []
        for i in range(src.shape[0]):
            ref.append(self.bert(reference_tensor[i], ref_segment_tensor[i]).last_hidden_state)

        # encoded = self.bert(src, seg)
        # 从word emb pooling到句子向量
        pooled_res = []
        for i in range(src.shape[0]):
            pooled_res.append(torch.mean(res[i].last_hidden_state, dim=1))

        gat = []
        for i in range(src.shape[0]):
            gat.append(self.GAT(pooled_res[i], rst[i])[:sent_num[i]])

        dec = []

        for i in range(src.shape[0]):
            enc = gat[i].view(-1,1,768).repeat(1, 128, 1)
            edu_polarity = polarity[i]
            edu_polarity_deleted =  [e for e in edu_polarity if e != 4]
            mapping = map[i][:len(edu_polarity_deleted)]
            max_mapping = max(mapping).item()
            enc = [ enc[i] for i in range(len(edu_polarity)) if edu_polarity[i] != 4]
            enc = torch.cat(enc, dim=0).reshape(mapping.shape[0], 128, -1)
            enc = torch_scatter.scatter(enc, mapping,dim=0,reduce="mean")
            reference = ref[i]
            reference = reference[:max_mapping+1]
            dec_res = self.decoder(reference, enc)
            fc_res = self.fc_out(dec_res)[:ref_num[i]]
            dec.append(fc_res)

        # sc_output = []
        # for i in range(src.shape[0]):
        #     sc_output.append( self.sc(pooled_res[i])[:sent_num[i]] )
        #
        # sc_output = torch.cat(sc_output, dim=0).reshape(-1, 5)
        return dec
