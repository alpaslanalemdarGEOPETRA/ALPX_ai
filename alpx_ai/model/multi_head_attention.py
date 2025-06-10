import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, mask=None):
        if key is None: key = query
        if value is None: value = query

        batch_size, seq_len, embed_dim = x.size()

        # Q, K, V estimate
        qkv = self.qkv_proj(x)  # (B, S, 3*E)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Her biri (B, S, H, D)

        # Başlıkları sıraya al
        q = q.transpose(1, 2)  # (B, H, S, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Skorları hesapla
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, S, S)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = attn @ v  # (B, H, S, D)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)  # (B, S, E)

        return self.out_proj(output)
