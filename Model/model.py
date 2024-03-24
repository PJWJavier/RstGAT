import torch
import torch.nn as nn
from Utils import str2embedding, str2encoder, str2target

class Model(nn.Module):
    """
    Pretraining models consist of three parts:
        - embedding
        - encoder
        - target
    """
    def __init__(self, args, embedding, encoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target

        if args.target in ["bert"] and args.tie_weights:
            self.target.mlm_linear_2.weight = self.embedding.word_embedding.weight


    def forward(self, src, tgt, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        loss_info = self.target(output, tgt)
        return loss_info


def build_model(args):
    """
    Build universial encoder representations models.
    The combinations of different embedding, encoder,
    and target layers yield pretrained models of different
    """

    embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
    encoder   = str2encoder[args.encoder](args)
    target    = str2target[args.target](args, len(args.tokenizer.vocab))
    model     = Model(args, embedding, encoder, target)

    return model

def load_model(model, model_path):
    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    return model

def save_model(model, model_path):
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
