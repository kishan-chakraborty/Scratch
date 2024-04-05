"""
Build the attention block for the diffusion model.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Self attention block of the attention architecture.
    """
    def __init__(self, n_heads: int, d_model: int, in_proj_bias: bool=True,
                 out_proj_bias: bool=True):
        super().__init__()

        # Create a large linear layer for the three matrix [Q, K, V]
        # shape of the weight matrix is (d_model, d_model * 3)
        self.in_proj = nn.Linear(d_model, d_model * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias)

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, x: torch.tensor, causal_mask:bool=False) -> torch.tensor:
        """
        Implementing the multihead attention mechanism

        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            causal_mask: Masking depending upon whether used in encoding or decoding.

        Returns:
            output tensor of shape (batch_size, seq_len, d_model)
        """
        input_shape = x.shape
        batch_size, n_seq, _ = input_shape

        # Shape of the tensor obtained after head split during multihead split.
        inter_shape = (batch_size, n_seq, self.n_heads, self.head_dim)

        # (batch_size, seq_len, 3*d_model) -> (batch_size, seq_len, 3*d_model)
        # -> Three tensors of shape (batch_size, seq_len, d_model) corresponding to
        # Querry, Key and Value matrix respectively.
        qry, key, val = self.in_proj(x).chunk(3, dim=-1)

        # ( batch_size, seq_len, d_model) -> (batch_size, n_seq, n_heads, head_dim)
        # -> (batch_size, n_heads, n_seq, head_dim)
        qry = qry.view(inter_shape).transpose(1, 2)
        key = key.view(inter_shape).transpose(1, 2)
        val = val.view(inter_shape).transpose(1, 2)

        # (batch_size, n_seq, n_heads, head_dim) @ (batch_size, n_seq, head_dim, n_heads)
        # -> (batch_size, n_heads, n_seq, n_seq)
        score = qry @ key.transpose(-1, -2)

        # Apply mask based on called during encoding or decoding
        if causal_mask:
            mask = torch.ones_like(score, dtype=torch.bool).triu(1)
            score.masked_fill(mask, -torch.inf)

        score /= math.sqrt(self.head_dim)
        score = F.softmax(score, dim=-1)

        # (batch_size, n_heads, n_seq, n_seq) @ (batch_size, n_heads, n_seq, head_dim)
        # -> (batch_size, n_heads, n_seq, head_dim)
        out = score @ val

        # (batch_size, n_heads, n_seq, head_dim) -> (batch_size, n_seq, n_heads, head_dim)
        out = out.transpose(1, 2)

        # (batch_size, n_seq, n_heads, head_dim) -> (batch_size, n_seq, n_heads*head_dim=d_model)
        out.view(input_shape)

        # Applying the output linear layer
        out = self.out_proj(out)

        return out
