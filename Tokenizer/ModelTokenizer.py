from Tokenizer.BaseTokenizer import BasicTokenizer, WordpieceTokenizer, Tokenizer
from Tokenizer.constants import *
from Tokenizer.CxGProcessor.CxGCore import CxGCore
from Tokenizer.Vocab import BERTVocab, CxGBERTVocab


class BertTokenizer(Tokenizer):
    def __init__(self, args):
        super().__init__(args, token_mode=CONST_TOKEN_MODE_WORD)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=args.do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=UNK_TOKEN)

    def tokenize(self, text) -> list:
        tokens, split_tokens = [], []
        if isinstance(text, str):
            text = [text]
        for ele in text:
            split_tokens = []
            for token in self.basic_tokenizer.tokenize(ele):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
            tokens.append(split_tokens)
        if len(text) == 1:
            return split_tokens
        return tokens


class CxGBertTokenizer(object):
    def __init__(self, args, visible=True):
        self.cxg = CxGCore(args)
        self.bert = BertTokenizer(args)
        self.bert_vocab = BERTVocab()
        self.cons_vocab = CxGBERTVocab()
        self.visible = visible

    def tokenize(self, text) -> dict:
        results = self.cxg.parse_text(text)
        # return results
        token_mask = {}
        for i, res in enumerate(results.values()):
            temp1 = {}
            token = res['token']
            for j, con_idx in enumerate(res["cons_idx"]):
                temp2 = {}
                con_start = res["cons_start"][j]
                con_end = res["cons_end"][j]
                part1 = [self.bert_vocab.word_w2i[w] for w in self.bert.tokenize(" ".join(token[:con_start]))]
                part2 = [con_idx]
                part3 = [self.bert_vocab.word_w2i[w] for w in self.bert.tokenize(" ".join(token[con_end:]))]
                temp2["idx"] = part1 + part2 + part3
                temp2["mask"] = [0]*len(part1) + [1] + [0]*len(part3)
                if self.visible:
                    token_ = self.bert.tokenize(" ".join(token[:con_start])) + [self.cons_vocab.cxg_i2c[con_idx]] + self.bert.tokenize(" ".join(token[con_end:]))
                    temp2["token_text"] = " ".join(token_)
                temp1[j] = temp2
            token_mask[res["text"]] = temp1
        return token_mask


if __name__  == '__main__':
    args      = ARG_TEST()
    tokenizer  = BertTokenizer(args)
    # tokenizer = CxGTokenizer(args)
    tokenizer1 = CxGBertTokenizer(args)
    sentences = ['The staff should be a bit more friendlyly.', "hello world"]
    # print(tokenier.tokenize(sentences))
    print(tokenizer.tokenize(sentences))
    res1 = tokenizer1.tokenize(sentences)
    print('')

