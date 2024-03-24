import json
import os

with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/special_tokens_map.json")), mode="r", encoding="utf-8") as f:
    special_tokens_map = json.load(f)

UNK_TOKEN = special_tokens_map["unk_token"]
CLS_TOKEN = special_tokens_map["cls_token"]
SEP_TOKEN = special_tokens_map["sep_token"]
MASK_WORD_TOKEN = special_tokens_map["mask_word_token"]
MASK_CXG_TOKEN = special_tokens_map["mask_cxg_token"]
PAD_TOKEN = special_tokens_map["pad_token"]

CONST_TOKEN_MODE_WORD = 'WORD'
CONST_TOKEN_MODE_CXG  = 'CXG'

try:
    class ARG_TEST(object):
        def __init__(self):
            self.word_vocab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/BERT/"))
            self.cxg_vocab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/CxGBERT"))
            self.do_lower_case = True
except:
    pass