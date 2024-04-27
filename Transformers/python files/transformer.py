"""
Assembling the built modules encoder and decoder in the transformer architecture.
"""
import torch
from torch import nn

from embedding import InputEmbedding, PositionalEncoding
from encoder import Encoder
from decoder import Decoder


class ProjectionLayer(nn.Module):
    """
    Implementing the projection layer. 
    This layer projects the output of decoder bock to vocab size.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Input tensor of shape [batch_size, max_seq_len, d_model]
        """
        # [batch_size, max_seq_len, d_model] -> [batch_size, max_seq_len, vocab_size]
        return torch.log_softmax(self.projection(x), dim=-1)

class Transformer:
    """
    Building the transformer architecture.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embedding: InputEmbedding,
                 tgt_embedding: InputEmbedding,
                 src_position: PositionalEncoding,
                 tgt_position: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_position = src_position
        self.tgt_position = tgt_position
        self.projection_layer = projection_layer

    def encode(self, x: torch.Tensor, mask: bool = False) -> torch.Tensor:
        """
        Encode the input sequence.
        """
        x = self.src_embedding(x)
        x = self.src_position(x)
        out = self.encoder(x, mask)
        return out

    def decode(self, encoder_output, src_mask, target, target_mask):
        """
        Decode the input sequence.
        """
        target = self.tgt_embedding(target)
        target = self.tgt_position(target)
        out = self.decoder(target, encoder_output, src_mask, target_mask)

        return self.projection_layer(out)
