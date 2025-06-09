import torch.nn as nn
from alpx_ai.model.multi_head_attention import MultiHeadAttention
from alpx_ai.model.positionwise_feedforward import PositionwiseFeedForward
from alpx_ai.model.positional_encoding import PositionalEncoding

class DecoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = PositionwiseFeedForward(embed_dim, ff_hidden_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # Masked self-attention
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # Encoder-Decoder attention
        enc_dec_attn_output = self.enc_dec_attn(x, enc_output, enc_output, memory_mask)
        x = self.norm2(x + self.dropout(enc_dec_attn_output))

        # Feedforward
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)

        return x
