from transformers import BertTokenizer
from transformers import BartTokenizer
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from Model.Utils import merge_str, pad_and_truncate, para_data_padding
from Model.Utils import Bart_data_padding as data_padding
from tqdm import trange
import re

maping = {"ignore":1,"rephrase":2,"ssplit":3,"delete":4,"dsplit":5}
def rst2graph(rst_res, max_sent_len = 25):
    all_graph = []
    for res in rst_res:
        begin = []
        end = []
        res = res.strip("[").strip("]")
        res_seg = res.split(",")
        res_seg = [seg.strip("'") for seg in res_seg]
        for seg in res_seg:
            segs = seg.split(";")
            direction = segs[1]
            if direction != "even":
                lefts = list(set(re.findall('\d+', segs[0])))
                right = list(set(re.findall('\d+', segs[2])))
                for l in lefts:
                    if int(l)< max_sent_len:
                        for r in right:
                            if int(r) < max_sent_len:
                                begin.append(int(l)-1)
                                end.append(int(r)-1)
                edge_begin = torch.tensor(begin)
                edge_end =  torch.tensor(end)
                edge_index = torch.cat([edge_begin,edge_end],dim=0).reshape(2,-1)
        all_graph.append(edge_index)
    return  all_graph
class efactdataset(Dataset):
    def __init__(self, args, fname, max_seq_len):
        self.maping = {"ignore":1,"rephrase":2,"ssplit":3,"delete":4,"dsplit":5}
        self.args = args
        self.bert_path = 'dataset/BERT/'
        # self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.tokenizer = args.tokenizer
        data = pd.read_csv(fname)
        if "train" in fname:
            data = data[:60000]
        elif "test" in fname:
            data = data[60000:]
            # data.loc[0] = [0,10,"vaccine for Ebola",10,"The first vaccine for Ebola was approved by the FDA in 2019 in the US, five years after the initial outbreak in 2014. To produce the vaccine, scientists had to sequence the DNA of Ebola, then identify possible vaccines, and finally show successful clinical trials. Scientists say a vaccine for COVID-19 is unlikely to be ready this year , although clinical trials have already started.","The first vaccine for Ebola took 5 years to be approved by the FDA. Scientists believe a vaccine for Covid-19 might not be ready this year.",	"['ignore', 'delete', 'ignore']",	28,	25,	2	,0,	"The first vaccine for Ebola was approved by the FDA in 2019 in the US, five years after the initial outbreak in 2014. [SEP] To produce the vaccine, scientists had to sequence the DNA of Ebola, then identify possible vaccines, and finally show successful clinical trials.[SEP]  Scientists say a vaccine for COVID-19 is unlikely to be ready this year , although clinical trials have already started." ,	"['(1:Nucleus=Joint:2,3:Nucleus=Joint:3) (1:Nucleus=span:1,2:Satellite=Elaboration:2)']"	,"[9, 23, 32]"	,"['1-2;even;3-3;even', '1-1;right;2-2;Elaboration']"]
        plain_article = data["article"].values
        edu_input = data["all_text"].values
        references = data["abstract"].values
        rst_res = data["parsing_res"].values
        rst_res = rst2graph(rst_res,self.args.max_sent_len)
        all_data = []
        max_length = 0
        for i in trange(0, len(data)):
            article = edu_input[i]
            abstract = references[i]
            rst_graph = rst_res[i]
            sentences = article.split('[SEP]')
            max_length = 0
            length = [len(sentence.split(" ")) for sentence in sentences]
            max_l = max(length)
            if max_l > max_length:
                max_length = max_l
            if len(sentences) >40:
                continue
            data = self.get_token_tensor(plain_article[i], article, abstract, max_seq_len, rst_graph, self.args.max_sent_len)
            if data:
                all_data.append(data)
        print('{} loaded, all {} sentences.'.format(fname,len(all_data)))
        # print(max_length)
        self.data=all_data

    def get_token_tensor(self,plain_article,  article, abstract, max_seq_len, rst_graph, max_sent_len = 25):

        # polarity = data_padding(max_seq_len, torch.tensor(polarity))

        # sent_num_mask = [[True] * max_seq_len] * 21 + [[False] * max_seq_len] * (max_sent_len - 21)

        article = article.split('[SEP]')
        article_sent_num = len(article)
        input_tokenizer =[['<s>'] + self.tokenizer.tokenize(sentence) + ['</s>'] for sentence in article]


        # input_sent_length = input_sent_length + (128-l) * [0]

        input_tensor = [data_padding(max_seq_len, torch.tensor(self.tokenizer.convert_tokens_to_ids(sentence))) for sentence in input_tokenizer]
        input_tensor = input_tensor[:article_sent_num]
        input_tensor = torch.cat(input_tensor,dim=0).reshape(len(input_tensor),-1)
        input_tensor = para_data_padding(max_seq_len, max_sent_len, input_tensor).int()

        rst_graph = [data_padding(max_seq_len, rst,max_sent_len + 2) for rst in rst_graph]
        rst_graph = torch.cat(rst_graph, dim=0).reshape(len(rst_graph), -1)

        segment = [data_padding(max_seq_len, torch.tensor([1] * len(sentence)),pad=0) for sentence in input_tokenizer]
        segment = segment[:article_sent_num]
        segment = torch.cat(segment, dim=0).reshape(len(segment), -1)
        segment_tensor = para_data_padding(max_seq_len,max_sent_len,segment).int()

        # input_tensor = data_padding(max_seq_len, torch.tensor(self.bertTokenizer.convert_tokens_to_ids(input_tokenizer)))
        references = abstract
        reference_tokenizer = ['<s>'] + self.tokenizer.tokenize(references) + ['</s>']
        reference_tensor = data_padding(max_seq_len, torch.tensor(self.tokenizer.convert_tokens_to_ids(reference_tokenizer)))
        ref_segment_tensor = data_padding(max_seq_len, torch.tensor([1] * len(reference_tokenizer)))

        # segment = [1] * len(input_tokenizer)
        # segment_tensor = data_padding(max_seq_len, torch.tensor(segment))
        data = {
            # 'input_tokenizer': input_tokenizer,
            "plain_article": plain_article,
            'sent_num': torch.tensor(article_sent_num),
            'rst': rst_graph,
            'map': map,
            # 'reference_tokenizer': reference_tokenizer,
            'input_tensor': input_tensor,
            'reference_tensor': reference_tensor,
            'ref_segment_tensor':ref_segment_tensor,
            "segment_tensor": segment_tensor
        }
        return data
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
