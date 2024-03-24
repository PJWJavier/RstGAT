from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from Model.Utils import merge_str, pad_and_truncate, data_padding
from Tokenizer.CxGProcessor.CxGCore import CxGCore

class GLUEDataset(Dataset):
    def __init__(self, args, fname, max_seq_len):
        self.args = args
        self.bert_path = 'dataset/BERT/'
        self.bertTokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.CxGCore = CxGCore(args)
        lines, lines_count = self.get_lines(fname)
        all_data = []
        if "train" in fname:
            # lines = lines[:60]
            for i in range(0, len(lines), lines_count):
                if lines_count == 2:
                    sentence1 = lines[i]
                    sentence2 = None
                    polarity = lines[i + 1]
                    data = self.get_token_tensor(sentence1,sentence2, polarity,max_seq_len)
                else:
                    sentence1 = lines[i]
                    sentence2 = lines[i + 1]
                    polarity = lines[i + 2]
                    data = self.get_token_tensor(sentence1, sentence2, polarity, max_seq_len)
                all_data.append(data)
        else:
            # lines = lines[:30]
            for i in range(0, len(lines), lines_count):
                if lines_count == 2:
                    sentence1 = lines[i]
                    sentence2 = None
                    polarity = lines[i + 1]
                    cxg1 = self.CxGCore.parse_text(sentence1)
                    data = self.get_token_tensor(sentence1, sentence2, polarity, max_seq_len)
                    data["cxg1"] = str(cxg1[0]["cons_idx"]) if len(cxg1[0]["cons_idx"]) != 0 else "None"
                    data["cxg2"] = "None"
                else:
                    sentence1 = lines[i]
                    sentence2 = lines[i + 1]
                    polarity = lines[i + 2]
                    cxg1 = self.CxGCore.parse_text(sentence1)
                    cxg2 = self.CxGCore.parse_text(sentence2)
                    data = self.get_token_tensor(sentence1, sentence2, polarity, max_seq_len)
                    data["cxg1"] = str(cxg1[0]["cons_idx"]) if len(cxg1[0]["cons_idx"]) != 0 else "None"
                    data["cxg2"] = str(cxg2[0]["cons_idx"]) if len(cxg2[0]["cons_idx"]) != 0 else "None"
                all_data.append(data)

        print('{} loaded, all {} sentences.'.format(fname,len(all_data)))
        self.data=all_data

    def get_lines(self, fname):
        lines = None
        lines_count = 2
        if 'CoLA' in fname:
            train = pd.read_csv(fname, sep='\t',header=None)
            train = train.iloc[:,[3,1]]
            train = train.values
            lines = np.array(train).reshape(train.shape[0] * train.shape[1],1)
            lines = [str(i[0]) for i in lines]
        elif 'MNLI' in fname:
            lines_count = 3
            train = pd.read_csv(fname, sep='\t' ,error_bad_lines=False)
            train = train.dropna()
            train.loc[train['gold_label'] == 'contradiction', 'gold_label'] = 0
            train.loc[train['gold_label']=='neutral','gold_label'] = 1
            train.loc[train['gold_label'] == 'entailment', 'gold_label'] = 2
            train = train[["sentence1", "sentence2", "gold_label"]]
            train = train.values
            lines = np.array(train).reshape(train.shape[0] * train.shape[1], 1)
            lines = [str(i[0]) for i in lines]
        elif 'MRPC' in fname:
            lines_count = 3
            train = pd.read_csv(fname, sep='\t' ,error_bad_lines=False)
            train = train.dropna()
            train = train[["#1 String", "#2 String", "Quality"]]
            train = train.values
            lines = np.array(train).reshape(train.shape[0] * train.shape[1], 1)
            lines = [str(i[0]) for i in lines]
        elif 'QNLI' in fname:
            lines_count = 3
            train = pd.read_csv(fname, sep='\t', error_bad_lines=False)
            train = train.dropna()
            train.loc[train['label'] == 'not_entailment', 'label'] = 0
            train.loc[train['label'] == 'entailment', 'label'] = 1
            train = train[["question", "sentence", "label"]]
            train = train.values
            lines = np.array(train).reshape(train.shape[0] * train.shape[1], 1)
            lines = [str(i[0]) for i in lines]
        elif 'QQP' in fname:
            lines_count = 3
            train = pd.read_csv(fname, sep='\t', error_bad_lines=False)
            train = train.dropna()
            train['is_duplicate'] =train['is_duplicate'].astype(np.int8)
            train = train[["question1", "question2", "is_duplicate"]]
            train = train.values
            lines = np.array(train).reshape(train.shape[0] * train.shape[1], 1)
            lines = [str(i[0]) for i in lines]
        elif 'RTE' in fname:
            lines_count = 3
            train = pd.read_csv(fname, sep='\t', error_bad_lines=False)
            train = train.dropna()
            train.loc[train['label'] == 'not_entailment', 'label'] = 0
            train.loc[train['label'] == 'entailment', 'label'] = 1
            train = train[["sentence1", "sentence2", "label"]]
            train = train.values
            lines = np.array(train).reshape(train.shape[0] * train.shape[1], 1)
            lines = [str(i[0]) for i in lines]
        elif 'SST-2' in fname:
            train = pd.read_csv(fname, sep='\t', error_bad_lines=False)
            train = train.dropna()
            train = train.values
            lines = np.array(train).reshape(train.shape[0] * train.shape[1], 1)
            lines = [str(i[0]) for i in lines]
        elif 'WNLI' in fname:
            lines_count = 3
            train = pd.read_csv(fname, sep='\t', error_bad_lines=False)
            train = train.dropna()
            train = train[["sentence1", "sentence2", "label"]]
            train = train.values
            lines = np.array(train).reshape(train.shape[0] * train.shape[1], 1)
            lines = [str(i[0]) for i in lines]
        elif 'STS-B' in fname:
            lines_count = 3
            train = pd.read_csv(fname, sep='\t', error_bad_lines=False)
            train = train.dropna()
            train['score'] = train['score'].apply(lambda x: round(x))
            train['score'] =train['score'].astype(np.int8)
            train = train[["sentence1", "sentence2", "score"]]
            train = train.values
            lines = np.array(train).reshape(train.shape[0] * train.shape[1], 1)
            lines = [str(i[0]) for i in lines]
        return lines, lines_count

    def get_token_tensor(self, sentence1, sentence2, polarity, max_seq_len):
        if sentence2 == None:
            all_text = sentence1
            polarity = int(polarity.lower().strip())
            all_tokenizer = ['[CLS]'] + self.bertTokenizer.tokenize(all_text) + ['[SEP]']
            all_index_tensor = data_padding(max_seq_len, torch.tensor(self.bertTokenizer.convert_tokens_to_ids(all_tokenizer)))
            segment = [1] * len(all_tokenizer)
            segment_tensor = data_padding(max_seq_len, torch.tensor(segment))
        else:
            all_text = sentence1 + ' ' + sentence2
            polarity = int(polarity.lower().strip())
            sentence1_tokenizer  = self.bertTokenizer.tokenize(sentence1)
            sentence2_tokenizer = self.bertTokenizer.tokenize(sentence2)
            all_tokenizer = ['[CLS]'] + sentence1_tokenizer + ['[SEP]'] + sentence2_tokenizer + ['[SEP]']
            all_index_tensor = data_padding(max_seq_len,
                                            torch.tensor(self.bertTokenizer.convert_tokens_to_ids(all_tokenizer)))
            segment = [1] * ( 2 + len(all_tokenizer) ) + [2] * ( 1 + len(all_tokenizer) )
            segment_tensor = data_padding(max_seq_len, torch.tensor(segment))

        data = {
            'all_text': all_text,
            'all_indices_tensor': all_index_tensor,
            'segment_tensor': segment_tensor,
            "polarity": polarity
        }
        return data
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
