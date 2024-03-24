import torch
import torch.nn as nn
import torch_scatter
from transformers import BertTokenizer, BartForConditionalGeneration
from Model.Core.BertForClassifier import BertForClassifierModel
from torch_geometric.nn import GATConv
from Model.Components.decoder import DecoderRNN

def data_padding(max_len,seq_tensor):
    x = torch.tensor([0.0]*max_len).cuda(seq_tensor.device)
    trunc = seq_tensor[:max_len]
    x[:trunc.shape[0]] = trunc
    return x

class BART_RST(nn.Module):
    def __init__(self, args):
        super(BART_RST, self).__init__()
        self.args = args
        self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base',cache_dir='./BART',local_files_only=True)
        self.GAT = GATConv(args.bert_dim, args.bert_dim, heads=1)
        self.max_sent_length = 128
        self.sc = nn.Linear(args.bert_dim, args.polarity_dim)
        self.decoder = nn.TransformerDecoderLayer(args.bert_dim, nhead=1)
        self.fc_out = nn.Linear(args.bert_dim, args.vocab_size)
    def forward(self, src=None,
                seg=None,
                reference_tensor = None,
                ref_segment_tensor = None,
                sent_num = None,
                rst = None,
                polarity = None,
                map = None,
                encoder_hidden = None,
                decoder_input_ids = None
                ):
        if decoder_input_ids is not None:
            outputs = self.bart(encoder_outputs=encoder_hidden, decoder_input_ids=decoder_input_ids)
            return outputs
        res = []
        for i in range(src.shape[0]):
            res.append(self.bart(src[i], seg[i]))
        ref = []
        # for i in range(src.shape[0]):
        #     ref.append(self.bart(reference_tensor[i], ref_segment_tensor[i]).encoder_last_hidden_state)

        # encoded = self.bert(src, seg)
        # 从word emb pooling到句子向量
        pooled_res = []
        for i in range(src.shape[0]):
            pooled_res.append(torch.mean(res[i].encoder_last_hidden_state, dim=1))

        gat = []
        for i in range(src.shape[0]):
            gat.append(self.GAT(pooled_res[i], rst[i])[:sent_num[i]])

        dec = []
        enc_all = []
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
            # reference = reference[:max_mapping+1]
            reference = reference_tensor[i][:max_mapping + 1]
            dec_res = self.bart(decoder_input_ids = reference, encoder_outputs = (enc, None, None))
            # dec_res = self.decoder(reference, enc)
            # fc_res = self.fc_out(dec_res)[:ref_num[i]]
            fc_res = dec_res.logits
            enc_all.append(enc)
            dec.append(fc_res)

        # sc_output = []
        # for i in range(src.shape[0]):
        #     sc_output.append( self.sc(pooled_res[i])[:sent_num[i]] )
        #
        # sc_output = torch.cat(sc_output, dim=0).reshape(-1, 5)
        return dec, enc_all
