from Model.Components.embedding import BertEmbedding
from Model.Core.BertModel import BertTarget
from Model.Encoder.TransformerEncoder import TransformerEncoder

str2embedding  = {"bert": BertEmbedding}
str2target     = {"bert": BertTarget}
str2encoder    = {"transformer": TransformerEncoder}

__all__ = []