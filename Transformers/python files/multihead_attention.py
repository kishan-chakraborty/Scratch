"""
Implementing the multihead attention layer of the transformer architecture.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    """
    Initialize multihead attention mechanism.

    Args:
        h: number of heads
        d_k: dimension of key and query vectors
        d_v: dimension of value vector
        d_emb: embedding dimension of each token.
        max_seq_len: maximum sequence length allowed.

        In the original paper d_k = d_v = d_emb = d_model = 512 and h = 8.
    """
    def __init__(self, h: int, d_k: int, d_v: int, d_emb: int) -> None:

        super().__init__()
        self.n_heads = h                    # No. of heads.
        self.d_k = d_k                      # key dimension.
        self.d_v = d_v                      # value dimension.
        self.d_model = h * d_v              # Model dimension.
        self.w_q = nn.Linear(d_emb, d_k*h)  # Query matrix.
        self.w_k = nn.Linear(d_emb, d_k*h)  # Key matrix.
        self.w_v = nn.Linear(d_emb, d_v*h)  # Value matrix.
        self.w_o = nn.Linear(self.d_model, self.d_model)  # Output matrix.

    def split_head(self, x: torch.tensor) -> torch.tensor:
        """
        Split the input tensor into multiple heads.

        Args:
            x: input tensor of shape (batch_size, max_seq_len, d_k*n_heads)

        Returns:
            Reshaped tensor of dimension (batch_size, n_heads, max_seq_len, d_k or d_v)
        """
        # (batch_size, max_seq_len, n_heads, d_k/d_v)
        x = x.view(x.shape[0], x.shape[1], self.n_heads, -1)
        # (batch_size, n_heads, max_seq_len, d_k/d_v)
        out = x.permute(0, 2, 1, 3)
        return out

    def forward(self, x, mask: bool) -> torch.tensor:
        """
        Forward pass of multihead attention.

        Args:
            x: input tensor of shape (batch_size, max_seq_len, d_k*n_heads)
            mask: whether to apply mask based on whether called during encoding/decoding.

        Returns:
            Reshaped tensor of dimension (batch_size, max_seq_len, d_model)
        """
        qs = self.w_q(x)       # Queries, shape: (batch_size, max_seq_len, d_k*n_heads)
        ks = self.w_k(x)       # Keys, shape: (batch_size, max_seq_len, d_k*n_heads)
        vs = self.w_v(x)       # Values, shape: (batch_size, max_seq_len, d_v*n_heads)

        qs = self.split_head(qs)   # (batch_size, n_heads, max_seq_len, d_k)
        ks = self.split_head(ks)   # (batch_size, n_heads, max_seq_len, d_k)
        vs = self.split_head(vs)   # (batch_size, n_heads, max_seq_len, d_v)

        # [batch_size, n_heads, max_seq_len, d_v]
        multihead_vals = MultiheadAttention.scaled_dotproduct_attention(qs, ks, vs, mask)

        # (batch_size, n_heads, max_seq_len, d_v) -> (batch_size, max_seq_len, d_model)
        multihead_vals = multihead_vals.transpose(1, 2).contiguous(). \
                            view(x.shape[0], -1, self.d_model)

        # (batch_size, max_seq_len, d_model)
        out = self.w_o(multihead_vals)
        return out

    @staticmethod
    def scaled_dotproduct_attention(qs: torch.tensor, ks: torch.tensor,
                 vs: torch.tensor, mask: bool) -> torch.Tensor:
        """
        Implementing scaled dot procduct attention.

        Calling scaled dot procduct attention.
            Args:
                qs: query matrix    [batch_size, n_heads, max_seq_len, d_k]
                ks: key matrix      [batch_size, n_heads, max_seq_len, d_k]
                vs: values  matrix  [batch_size, n_heads, max_seq_len, d_v]
                mask: whether to apply mask based on whether called during encoding/decoding.

            Returns:
                Calculated attention weights. [batch_size, n_heads, max_seq_len, d_v]
        """
        # [batch_size, n_heads, max_seq_len, d_k] @ [batch_size, n_heads, d_k, max_seq_len]
        # --> [batch_size, n_heads, max_seq_len, max_seq_len]
        score_mat = qs @ ks.permute(0, 1, 3, 2)
        score_mat_scaled = score_mat / math.sqrt(len(ks[0]))

        if mask:
            # Replace all the zeros by -10^9.
            score_mat_scaled = score_mat_scaled.masked_fill(mask == 0, -1e9)

        attention = F.softmax(score_mat_scaled, -1)
        # [batch_size, n_heads, max_seq_len, max_seq_len] @ [batch_size, n_heads, max_seq_len, d_v]
        # --> [batch_size, n_heads, max_seq_len, d_v]
        out =  attention @ vs
        return out
