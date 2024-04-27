"""
Implementing the embedding layer and positional encoding layer of the transformer architecture.

Embedding layer maps an integer to a vector in higher dimension.
These vectors are learned during training the transformer architecture.

Positional encoding layer adds positional information to the x sequence of tokens.
This is a fixed layer which is not learned during traninng.
"""

import math
import torch
from torch import nn

class InputEmbedding(nn.Module):
    """
    Implementing the embedding layer of the transformer architecture.

    Args:
        vocab_size: Size of the vocabulary
        embed_dim: Embedding Dimension.
    """
    def __init__(self, vocab_size:int, embed_dim:int=512) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x: torch.Tensor):
        """
        Implementing the forward pass.
        """
        out = self.embedding(x) * math.sqrt(self.d_model)
        return out

class PositionalEncoding(nn.Module):
    """
    Implementing the positional encoding layer of the transformer architecture.

    Args:
        d_model: Embedding dimension.
        max_seq: Maximum sequence length.
        encod_pos: Positional information of the x sequence.
    """
    def __init__(self, d_model:int, max_seq:int) -> None:
        super().__init__()
        self.max_seq = max_seq
        self.d_model = d_model

        # From the notebook we can see that denominator of both odd and even dim are same.
        denominator = 1000**(torch.arange(0, self.d_model, 2)/self.d_model)

        # Creating the position vector corresponding to max seq len. (max_seq, 1)
        positions = torch.arange(0, self.max_seq, dtype=torch.float).unsqueeze(1)
        # Sinusoidal positional encoding shape (max_seq, d_model/2)
        even_pe = torch.sin(positions/denominator)
        odd_pe = torch.cos(positions/denominator)

        stack = torch.stack([even_pe, odd_pe], dim=2) # (max_seq, d_model/2, 2)
        self.encoded_pos = torch.flatten(stack, start_dim=1, end_dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of positional encoding.

        Args:
            x: Input embedding tensor

        Returns:
            Positional information of the x sequence.
        """
        out = x + self.encoded_pos.requires_grad(False)
        return out

class LayerNormalization(nn.Module):
    """
    Implementing the batch normalization layer of the transformer architecture.

    Args:
        d_model: Embedding dimension.
        gamma: Learnable parameter (multiply)
        beta: Learnable parameter (add)
    """
    def __init__(self, d_model:int, esp: float=1e-5) -> None:
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(1, d_model))
        self.beta = nn.Parameter(torch.zeros(1, d_model))
        self.esp = esp

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embedding tensor shape (batch_size, max_seq_len, d_model)

        Return:
            Normalized x
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = var.sqrt()
        y = (x - mean) / (std + self.esp)
        out = self.gamma * y + self.beta
        return out
