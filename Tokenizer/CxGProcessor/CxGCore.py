from Tokenizer.CxGProcessor.Loader import Loader
from Tokenizer.CxGProcessor.Encoder import Encoder
from Tokenizer.CxGProcessor.Parser import Parser
from Tokenizer.constants import ARG_TEST


class CxGCore(object):
    def __init__(self, args):
        self.args = args
        self.Loader = Loader(args)
        self.Encoder = Encoder()
        self.Parser = Parser(self.Loader, self.Encoder, workers=None)

    def parse_text(self, text):
        if isinstance(text, str):
            text = [text]
        tokens = self.Loader.tokenize(text)
        lines = self.Loader.load_text(text)
        results = self.Parser.parse_lines(lines)
        # return results
        results_ = {}
        for i, res in enumerate(results):
            temp = {}
            temp["text"] = text[i]
            temp["token"] = tokens[i]
            temp["cons_idx"] = res[0]
            temp["cons_start"] = res[1]
            temp["cons_end"] = res[2]
            results_[i] = temp
        return results_

    def parse_file(self, file):
        lines = self.Loader.load_from_file(file)
        results = self.Parser.parse_lines(lines)
        return results


if __name__ == "__main__":
    import os
    args = ARG_TEST()
    cxg = CxGCore(args)
    text = ["Bill gave Wendy a hand.", "the game is unfriendly", "As I see , the staff should be friendly .", "in my opinion, the staff should be fired", "Anywhere else, the price would be 3x as high!"]
    results = cxg.parse_text(text)
    # file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../corpus/bert_senti.txt"))
    # results_ = cxg.parse_file(file)


