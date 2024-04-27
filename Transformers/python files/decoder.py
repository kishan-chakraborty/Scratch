"""
Implementaion of the decoder block.
"""

import torch
from torch import nn

from multihead_attention import MultiheadAttention
from feed_forward import FeedForward
from residual_layer import ResidualConnection
from embedding import LayerNormalization

class Decoder(nn.Module):
    """
    Args:
        self_attention: Multihead Attention
        cross_attention: Multihead Attention
        feed_forward: Feed Forward layer
        dropout: dropout rate
    """
    def __init__(self, self_attention: MultiheadAttention,
                 cross_attention: MultiheadAttention,
                 feed_forward: FeedForward,
                 dropout: float=None) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward

        if dropout:
            self.dropout = nn.Dropout(dropout)

        d_model = self.self_attention.d_model
        self.res_conns = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: bool, tgt_mask: bool) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, max_seq_len, d_model]
            encoder_output: Encoder output tensor of shape [batch_size, max_seq_len, d_model]
            source_mask: Mask applied to the encoder
            tgt_mask: Mask applied to the decoder

        Returns:
            output tensor of shape [batch_size, max_seq_len, d_model]
        """
        x = self.res_conns[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.res_conns[1](x, lambda x: self.cross_attention(x, encoder_output,
                                                                encoder_output, src_mask))
        x = self.res_conns[2](x, self.feed_forward)

        return x

class DecoderBlock(nn.Module):
    """
    Args:
        layers: List of decoder blocks
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        d_model = self.layers[0].d_model
        self.norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: bool, tgt_mask: bool) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, max_seq_len, d_model]
            encoder_output: Encoder output tensor of shape [batch_size, max_seq_len, d_model]
            source_mask: Mask applied to the encoder
            tgt_mask: Mask applied to the decoder

        Returns:
            output tensor of shape [batch_size, max_seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)
