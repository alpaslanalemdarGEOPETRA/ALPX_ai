import torch
import torch.nn as nn
from alpx_ai.model.multi_head_attention import MultiHeadAttention
from alpx_ai.model.positionwise_feedforward import PositionwiseFeedForward

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = PositionwiseFeedForward(embed_dim, ff_hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention + residual + norm
        attn_output = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feedforward + residual + norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
