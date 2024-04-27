"""
Implementing the residual connection in the transformer architecture.

    normalize( [Input Tensor] + [sublayer output] )

    Input Tensor: [embedding + positional encoding] OR output of sublayer
    eublayer: Feedforward or Multihead attention

"""

import torch
from torch import nn

from embedding import LayerNormalization

class ResidualConnection(nn.Module):
    """
    Implementing the residual connection in the transformer architecture.

    Args:
        d_model: dimension of the model = 512.
        dropout: dropout rate = 0.1.    
    """
    def __init__(self, d_model: int, dropout: float = None) -> None:
        super().__init__()

        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, max_seq_len, d_model]
            sublayer: Multihead attention or feedforward layer.

        output:
            normalize(x + sublayer(x))
        """
        x = x + self.dropout(sublayer(x))
        out = self.norm(x)

        return out
