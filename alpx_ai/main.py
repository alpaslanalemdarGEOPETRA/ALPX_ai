from alpx_ai.model.multi_head_attention import MultiHeadAttention

import torch

if __name__ == "__main__":
    attention = MultiHeadAttention(embed_dim=512, num_heads=8)
    x = torch.rand(2, 10, 512)  # (batch_size, sequence_length, embed_dim)
    out = attention(x, x, x)
    print("Output shape:", out.shape)
