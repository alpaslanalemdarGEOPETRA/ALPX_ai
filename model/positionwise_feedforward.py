# alx_ai/modules/positionwise_feedforward.py

import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
