"""
Build the clip encoder module. 
This is to create an embedding for the input prompt.
"""

import sys
import torch
from torch import nn

sys.path.append(r"C:\Users\darsh\Desktop\Projects\Scratch\StableDiffusion\VariationalAutoEncoder")

from attention import SelfAttention

class ClipEmbedding(nn.Module):
    """
    Build the embedding for the input prompt.

    Args:
        vocab_size: size of the vocabulary
        embed_dim: embedding dimension
        n_tokens: Maximum seq length
    """

    def __init__(self, vocab_size: int, embed_dim: int, n_tokens: int) -> None:
        super().__init__()

        # Embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Positional embedding, a learnable paremeter unlike the standard transformer architecture.
        self.positional_embedding = nn.Parameter(torch.zeros(n_tokens, embed_dim))

    def forward(self, x):
        """
        Forward pass of the embedding layer.

        Args:
            x: input tensor of shape (batch_size, max_length)

        Returns:
            out: output tensor of shape (batch_size, max_length, embed_dim)
        """
        out = self.token_embedding(x) + self.positional_embedding
        return out

class ClipLayer(nn.Module):
    """
    Clip encoder layer.
    Implementing the encoder module of a transformer architecture.

    Args:
        n_head: number of attention heads
        embed_dim: embedding dimension of each token
    """

    def __init__(self, n_head: int, embed_dim: int) -> None:
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(n_head, embed_dim)
        self.linear1 = nn.Linear(embed_dim, 4*embed_dim)
        self.linear2 = nn.Linear(4*embed_dim, embed_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Implementing the forward pass of encoder module of the transformer architecturem (Clip).
        This is little bit different from the original transformer architecture.

        args:
            x: input tensor of shape (batch_size, max_length, embed_dim)
        """
        residue = x

        # Self attention
        x = self.layer_norm1(x)
        x = self.attn(x, causal_mask = True)
        x += residue

        residue = x

        x = self.layer_norm2(x)
        x = self.linear1(x)
        x = x * torch.sigmoid(1.782*x)  # QuickGELU activation function
        x = self.linear2(x)

        out = residue + x
        return out

class Clip(nn.Module):
    """
    Clip encoder module.
    """
    def __init__(self) -> None:
        super().__init__()
        self.embedding = ClipEmbedding(49488, 768, 77)

        self.layers =nn.Module([
            ClipLayer(12, 768) for _ in range(12)
        ])

        self.layernorm = self.layernorm(768)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        The forward pass of the clip encoder module.

        Args:
            x: input tensor of shape (batch_size, max_length)
        """

        tokens = x.type(torch.long)

        # (batch_size, max_length) ->  (batch_size, max_length, embed_dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output
