import torch
import torch.nn as nn
import torch_scatter
from transformers import BertTokenizer, BartForConditionalGeneration
from Model.Core.BertForClassifier import BertForClassifierModel
from torch_geometric.nn import GATConv
from Model.Components.GAT import GAT
from Model.Components.decoder import DecoderRNN
import torch.nn.functional as F

def data_padding(max_len,seq_tensor):
    x = torch.tensor([0.0]*max_len).cuda(seq_tensor.device)
    trunc = seq_tensor[:max_len]
    x[:trunc.shape[0]] = trunc
    return x

def mask_select(seq,mask, bert_dim):
    mask = mask == 1
    seg_mask = mask.repeat(bert_dim, 1, 1).permute(1,2,0)

    seq = torch.masked_select(seq, seg_mask).view(-1, bert_dim)
    return seq

class RSTBART_AS(nn.Module):
    def __init__(self, args):
        super(RSTBART_AS, self).__init__()
        self.args = args
        self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn',cache_dir='./BART_large',local_files_only=True)
        # self.GAT = GATConv(args.bert_dim, args.bert_dim, heads=1)
        self.GAT = GAT(args.max_seq_len, args.bert_dim, args.device)

        self.max_sent_length = 128
        self.sc = nn.Linear(args.bert_dim, args.polarity_dim)
        self.decoder = nn.TransformerDecoderLayer(args.bert_dim, nhead=1)
        self.fc_out = nn.Linear(args.bert_dim, args.vocab_size)
    def forward(self, input=None,
                seg=None,
                reference_tensor = None,
                ref_segment_tensor = None,
                sent_num = None,
                rst = None,
                encoder_hidden = None,
                decoder_input_ids = None,
                mode = None
                ):
        if mode == "generate":
            # outputs = self.bart(decoder_input_ids=decoder_input_ids,encoder_outputs=(encoder_hidden, None, None))
            outputs = self.bart.generate(input_ids=None, attention_mask=None,max_length = 25,
                                       repetition_penalty=1.0, num_beams=8, encoder_outputs=encoder_hidden)

            return outputs
        else:
            res = []
            for i in range(input.shape[0]):
                res.append(self.bart(input[i], seg[i]).encoder_last_hidden_state)
            # encoded = self.bert(src, seg)
            # 从word emb pooling到句子向量
            # self.bart.generate(input[0],max_length=64,num_beams=5,do_sample=False,no_repeat_ngram_size=2)
            # pooled_res = []
            # for i in range(input.shape[0]):
            #     # pooled_res.append(   torch.mean(res[i].encoder_last_hidden_state, dim=1))
            #     pooled_res.append(res[i].encoder_last_hidden_state)
            gat = []
            for i in range(input.shape[0]):
                rst_mask = rst[i] != self.args.max_sent_len + 2
                single_rst = torch.masked_select(rst[i], rst_mask).view(2, -1)
                gat_res = self.GAT(res[i],seg[i], single_rst)[:sent_num[i]]
                gat_res = mask_select(gat_res, seg[i][:sent_num[i]],self.args.bert_dim)

                gat.append(gat_res)

            dec = []
            enc_all = []
            for i in range(input.shape[0]):
                enc = gat[i]
                enc = torch.unsqueeze(enc, 0)
                reference = reference_tensor[i]
                reference = torch.unsqueeze(reference, 0)
                dec_res = self.bart(decoder_input_ids = reference, encoder_outputs = (enc, None, None))
                # dec_res = self.decoder(reference, enc)
                # fc_res = self.fc_out(dec_res)[:ref_num[i]]
                fc_res = dec_res.logits
                enc = F.pad(enc,(0,0,0,1024-enc.shape[1],0,0), mode='constant', value=0)
                enc_all.append(enc)
                dec.append(fc_res)

            # sc_output = []
            # for i in range(src.shape[0]):
            #     sc_output.append( self.sc(pooled_res[i])[:sent_num[i]] )
            #
            # sc_output = torch.cat(sc_output, dim=0).reshape(-1, 5)
            if mode == "train":
                return dec
            if mode == "eval":
                return enc_all
