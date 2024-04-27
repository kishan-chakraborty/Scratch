"""
Implementing the point wise feed forward layer of the transformer architecture.

    The formula is given by max(0, xW1 + b1)W2 + b2
"""

import torch
from torch import nn

class FeedForward(nn.Module):
    """
    Args:
        d_model: dimension of the model = 512.
        d_ff: dimension of the feed forward layer = 2048.
        dropout: dropout rate = 0.1.
    """
    def __init__(self, d_model: int, d_ff: int, dropout:float = None) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, max_seq_len, d_model)

        Returns:
            output tensor of shape (batch_size, max_seq_len, d_model)
        """
        x = self.linear1(x)
        x = torch.relu(x)
        if self.dropout:
            x = self.dropout(x)
        out = self.linear2(x)

        return out
