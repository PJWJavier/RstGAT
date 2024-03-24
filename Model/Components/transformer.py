import torch.nn as nn
from .layernorm import LayerNorm
from .ffn import PositionwiseFeedForward
from .attention import MultiHeadedAttention

class TransformerBlock(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, args):
        super(TransformerBlock, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = args.hidden_size // args.heads_num

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias, with_scale = with_scale
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        # Feed forward layer.
        self.feed_forward = PositionwiseFeedForward(
            args.hidden_size, args.feedforward_size, args.hidden_act, has_bias
        )
        self.dropout_2 = nn.Dropout(args.dropout)

        self.layer_norm_1 = LayerNorm(args.hidden_size)
        self.layer_norm_2 = LayerNorm(args.hidden_size)

    def forward(self, hidden, mask, position_bias = None, has_residual_attention=False, prev_attn=None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        if self.layernorm_positioning == "post":
            inter, prev_attn_out = self.self_attn(hidden, hidden, hidden, mask, position_bias, has_residual_attention, prev_attn)
            inter = self.dropout_1(inter)
            inter = self.layer_norm_1(inter + hidden)
            output = self.dropout_2(self.feed_forward(inter))
            output = self.layer_norm_2(output + inter)
        else:
            inter = self.layer_norm_1(hidden)
            inter, prev_attn_out = self.self_attn(inter, inter, inter, mask, position_bias, has_residual_attention, prev_attn)
            inter = self.dropout_1(inter)
            hidden = hidden + inter
            output = self.layer_norm_2(hidden)
            output = self.dropout_2(self.feed_forward(output)) + hidden
        return output, prev_attn_out