"""
Implementing a single encoder block of transforner architecture

    
Formula: 
    layer1_output = [embedding+positional encoding] + [multihead output]
    normalize1 = normalize(layer1_output)

    layer2_output = [normalized1] + [feed_forward(normalized1) --> output]
    normalized(layer2_output)
"""

import torch
from torch import nn

from multihead_attention import MultiheadAttention
from feed_forward import FeedForward
from residual_layer import ResidualConnection
from embedding import LayerNormalization

class EncoderBlock(nn.Module):
    """
    Implementing a single encoder block of transforner architecture.

    Args:
        attention_block: Multihead Attention block.
        feed_forward_block: Feed forward block.
        dropout: dropout rate.
    """

    def __init__(self, attention_block: MultiheadAttention, feed_forward_block: FeedForward,
                 dropout: float=None) -> None:
        super().__init__()
        self.attention_block = attention_block
        self.feedforward_block = feed_forward_block
        self.d_model = attention_block.d_model
        self.residual_connetions = nn.ModuleList([ResidualConnection(self.d_model, dropout)
                                                   for _ in range(2)])

    def forward(self, x: torch.Tensor, source_mask: bool = False) -> torch.Tensor:
        """
        x: Input tensor to the encoder block shape [batch_size, max_seq_len, d_model]
        source_mask: If masking to be applied. Thsi is applied to the empty padding or 
                upper triangular values during decoding.

        Returns:
            Encoded data. shape [batch_size, max_seq_len, d_model]
        """

        x = self.residual_connetions[0](x, lambda x: self.attention_block(x, x, x, source_mask))
        x = self.residual_connetions[1](x, self.self.feedforward_block)
        return x

class Encoder(nn.Module):
    """
    Implementing the encoder module of the transformer architecture.

    Args:
        layers: List of encoder blocks. original paper has 6 encoder blocks.
    """
    def __init__(self, layers: nn.ModuleList):
        super().__init__()

        self.layers = layers
        d_model = self.layers[0].d_model
        self.norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, mask: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, max_seq_len, d_model]
            mask: Bool mask to be applied to empty padding or upper
                    triangular values during decoding.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
