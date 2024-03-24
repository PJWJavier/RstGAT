from transformers import BertTokenizer
from transformers import BartTokenizer
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from Model.Utils import merge_str, pad_and_truncate, data_padding, para_data_padding
from Tokenizer.CxGProcessor.CxGCore import CxGCore
import re

maping = {"ignore":1,"rephrase":2,"ssplit":3,"delete":4,"dsplit":5}
def rst2graph(rst_res, max_seq_len, max_sent_len = 25):
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
def edu_polarity(input, edu_polarity,references_num, max_sent_len=25):
    ret = []
    edu_map = []
    for i in range(len(input)):
        edu_ret = []
        sent = input[i]
        polarity = edu_polarity[i]
        sentences = sent.split(".")

        polarity = polarity.strip("[").strip("]")
        polarity = polarity.split(",")
        sent_num = len(polarity) if len(polarity) <= max_sent_len else max_sent_len
        sentences = sentences[:sent_num]
        polarity = [p.strip("'").strip(" \'") for p in polarity]
        polarity = [maping[p] for p in polarity]
        polarity = polarity[:max_sent_len]
        sent, pre, map = 0, 0, []
        for j in range(len(sentences)):
            # edus= sentences[j].split("[SEP]")
            # edus = [edu  for edu in edus if len(edu) > 3]
            cnt = sentences[j].count("[SEP]")
            edu_ret += cnt * [polarity[j]]
            if polarity[j] != 4 and cnt != 0:
                map += cnt * [sent]
                sent = sent + 1
        edu_ret += 1 * [polarity[-1]]
        edu_ret_deleted = [edu for edu in edu_ret if edu != 4]
        if polarity[-1] != 4: map += 1 * [sent-1]
        if len(map) == 0:
            map += 1 * [0]
        try:
            while(map[-1] + 1 < references_num[i]) :
                map += 1 * [map[-1] + 1]
            if len(map) < references_num[i]:
                map += (references_num[i] - len(map)) * [references_num[i] -1]
            assert map[-1] + 1 == references_num[i]
            while len(map) > len(edu_ret_deleted):
                edu_ret += 1 * [1]
                edu_ret_deleted += 1 * [1]
            assert len(map) == len(edu_ret_deleted)
        except:
            print(i)
        edu_map.append(map)
        ret.append(edu_ret)
    return ret, edu_map
class WikiTSDaraset(Dataset):
    def __init__(self, args, fname, max_seq_len, max_sent_len = 25):
        self.maping = {"ignore":1,"rephrase":2,"ssplit":3,"delete":4,"dsplit":5}
        self.args = args
        self.bert_path = 'dataset/BERT/'
        # self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',cache_dir='./BART',local_files_only=True)
        data = pd.read_csv(fname)
        if "train" in fname:
            data = data[:500]
        elif "test" in fname:
            data = data[:20]
            # data.loc[0] = [0,10,"vaccine for Ebola",10,"The first vaccine for Ebola was approved by the FDA in 2019 in the US, five years after the initial outbreak in 2014. To produce the vaccine, scientists had to sequence the DNA of Ebola, then identify possible vaccines, and finally show successful clinical trials. Scientists say a vaccine for COVID-19 is unlikely to be ready this year , although clinical trials have already started.","The first vaccine for Ebola took 5 years to be approved by the FDA. Scientists believe a vaccine for Covid-19 might not be ready this year.",	"['ignore', 'delete', 'ignore']",	28,	25,	2	,0,	"The first vaccine for Ebola was approved by the FDA in 2019 in the US, five years after the initial outbreak in 2014. [SEP] To produce the vaccine, scientists had to sequence the DNA of Ebola, then identify possible vaccines, and finally show successful clinical trials.[SEP]  Scientists say a vaccine for COVID-19 is unlikely to be ready this year , although clinical trials have already started." ,	"['(1:Nucleus=Joint:2,3:Nucleus=Joint:3) (1:Nucleus=span:1,2:Satellite=Elaboration:2)']"	,"[9, 23, 32]"	,"['1-2;even;3-3;even', '1-1;right;2-2;Elaboration']"]

        input = data["complex"].values
        references = data["simple"].values
        references_num = [len(ref.split("<s>"))for ref in references]
        polarity = data["labels"].values
        edu_input = data["all_text"].values
        rst_res = data["parsing_res"].values

        rst_res = rst2graph(rst_res,max_seq_len)
        polarity, edu_map = edu_polarity(edu_input,polarity,references_num)
        all_data = []
        for i in range(0, len(data)):
            paragh1 = input[i].replace("<s>",'[SEP]')
            paragh2 = references[i].replace("<s>",'[SEP]')
            ref_num = references_num[i]
            polarity1 = polarity[i]
            rst = rst_res[i]
            edus = edu_input[i]
            if len(edus.split('[SEP]')) >25:
                continue
            map = edu_map[i]
            if -1 in map:
                continue
            data = self.get_token_tensor(paragh1, paragh2,polarity1, max_seq_len, edus, rst,ref_num, map)
            if data:
                all_data.append(data)
        print('{} loaded, all {} sentences.'.format(fname,len(all_data)))
        self.data=all_data

    def get_token_tensor(self, sentence1, sentence2, polarity, max_seq_len, edu_input, rst, ref_num, map, max_sent_len = 25):

        sent_num = len(polarity) if len(polarity) <= max_sent_len else max_sent_len
        polarity = polarity[:max_sent_len]
        # polarity = data_padding(max_seq_len, torch.tensor(polarity))

        # sent_num_mask = [[True] * max_seq_len] * 21 + [[False] * max_seq_len] * (max_sent_len - 21)

        sentence1 = edu_input.split('[SEP]')
        input_tokenizer =[['[CLS]'] + self.tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentence1]
        split = [i for i,x in enumerate(input_tokenizer) if x == '[SEP]']
        sent = 1
        pre, input_sent_length = 0, []
        for s in split:
            input_sent_length += (s-pre) * [sent]
            pre = s
            sent = sent +1
        input_sent_length += (max_seq_len - len(input_sent_length)) * [0]
        if len(input_sent_length) > max_seq_len:
            return None
            pass
        input_sent_length = data_padding(max_seq_len, torch.tensor(input_sent_length))
        # input_sent_length = input_sent_length + (128-l) * [0]

        input_tensor = [data_padding(max_seq_len, torch.tensor(self.tokenizer.convert_tokens_to_ids(sentence))) for sentence in input_tokenizer]
        input_tensor = input_tensor[:sent_num]
        input_tensor = torch.cat(input_tensor,dim=0).reshape(len(input_tensor),-1)
        input_tensor = para_data_padding(max_seq_len, max_sent_len, input_tensor).int()

        edu_input_tensor = edu_input.split('[SEP]')
        edu_num = len(edu_input_tensor)
        edu_input_tokenizer = [['[CLS]'] + self.tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in edu_input_tensor]
        edu_input_tensor = [data_padding(max_seq_len, torch.tensor(self.tokenizer.convert_tokens_to_ids(sentence))) for
                            sentence in edu_input_tokenizer]
        edu_input_tensor = edu_input_tensor[:edu_num]
        edu_input_tensor = torch.cat(edu_input_tensor, dim=0).reshape(edu_num, -1)
        edu_input_tensor = para_data_padding(max_seq_len, max_sent_len, edu_input_tensor).int()

        segment = [data_padding(max_seq_len, torch.tensor([1] * len(sentence))) for sentence in input_tokenizer]
        segment = segment[:sent_num]
        segment = torch.cat(segment, dim=0).reshape(len(segment), -1)
        segment_tensor = para_data_padding(max_seq_len,max_sent_len,segment).int()

        # input_tensor = data_padding(max_seq_len, torch.tensor(self.bertTokenizer.convert_tokens_to_ids(input_tokenizer)))
        sentence2 = sentence2.split('[SEP]')
        reference_tokenizer = [['[CLS]'] + self.tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentence2]
        reference_tensor = [data_padding(max_seq_len, torch.tensor(self.tokenizer.convert_tokens_to_ids(sentence))) for
                            sentence in reference_tokenizer]
        reference_tensor = reference_tensor[:ref_num]
        reference_tensor = torch.cat(reference_tensor, dim=0).reshape(ref_num, -1)
        reference_tensor = para_data_padding(max_seq_len, max_sent_len, reference_tensor).int()

        ref_segment = [data_padding(max_seq_len, torch.tensor([1] * len(sentence))) for sentence in reference_tokenizer]
        ref_segment = ref_segment[:ref_num]
        ref_segment = torch.cat(ref_segment, dim=0).reshape(len(ref_segment), -1)
        ref_segment_tensor = para_data_padding(max_seq_len, max_sent_len, ref_segment).int()
        # segment = [1] * len(input_tokenizer)
        # segment_tensor = data_padding(max_seq_len, torch.tensor(segment))
        data = {
            # 'input_tokenizer': input_tokenizer,
            'sent_num': sent_num,
            'ref_num': ref_num,
            'rst': rst,
            'map': map,
            'input_sent_length': input_sent_length,
            # 'reference_tokenizer': reference_tokenizer,
            'input_tensor': input_tensor,
            'edu_input_tensor': edu_input_tensor,
            'reference_tensor': reference_tensor,
            'ref_segment_tensor':ref_segment_tensor,
            'polarity': polarity,
            "segment_tensor": segment_tensor
        }
        return data
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
