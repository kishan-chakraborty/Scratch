"""
    Building the decoder module of variational encoder and decoder architecture.
"""
import torch
from torch import nn
import torch.nn.functional as F

from attention import SelfAttention

class VAEAttentionBlock(nn.Module):
    """
        Self attention block for the decoder module.
    """
    def __init__(self, in_channels:int):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.attention = SelfAttention(1, in_channels)

    def forward(self, x:torch.tensor) -> torch.tensor:
        """
        Forward pass of the attention block.

        Args:
            x: input tensor of shape (batch_size, features, height, width)

        Returns:
            Output tensor of shape (batch_size, in_channels, height, width)
        """
        n_batch, n_feat, height, width = x.shape

        # (batch_size, features, height, width) -> (batch_size, features, )
        x = x.view(n_batch, n_feat, height * width)

        # To apply self attentio to every pixel reshape the input tensor just like transformer.
        # (batch_size, features, height*width) -> (batch_size, height*width, features)
        x = x.transpose(-1, -2)

        # (batch_size, height*width, features) -> (batch_size, height*width, features)
        x = self.attention(x)

        # (batch_size, height*width, features) -> (batch_size, features, height*width)
        x = x.transpose(-1, -2)

        # (batch_size, height*width, features) -> (batch_size, features, height, width)
        x = x.view(n_batch, n_feat, height, width)


class VAEResidualBlock(nn.Module):
    """
        Residual block for the decoder module.
    """
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x:torch.tensor) -> torch.tensor:
        """
        Forward pass of the residual block.

        Args:
            x: input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        residue = x

        x = self.group_norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.group_norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        out = self.residual_layer(residue) + x
        return out


class Decoder(nn.Module):
    """
        Building the decoder module of variational encoder and decoder architecture.
    """
    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(4, 512, kernel_size=3, padding=0, stride=1),

            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAEResidualBlock(512, 512),

            VAEAttentionBlock(512),

            # (batch_size, 512, height/8, width/8) -> # (batch_size, 512, height/8, width/8)
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 128, height/4, width/4)
            nn.Upsample(scale_factor=2),

            # (batch_size, 128, height/4, width/4) -> (batch_size, 128, height/4, width/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),

            # (batch_size, 512, height/4, width/4) -> # (batch_size, 512, height/4, width/4)
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/2, width/2)
            nn.Upsample(scale_factor=2),

            # (batch_size, 128, height/4, width/4) -> (batch_size, 128, height/4, width/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),

            # (batch_size, 512, height/4, width/4) -> # (batch_size, 256, height/4, width/4)
            VAEResidualBlock(512, 256),
            VAEResidualBlock(256, 256),
            VAEResidualBlock(256, 256),

            # (batch_size, 512, height/2, width/2) -> (batch_size, 512, height, width)
            nn.Upsample(scale_factor=2),

            # (batch_size, 256, height, width) -> (batch_size, 256, height, width)
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),

            ## (batch_size, 256, height, width) -> (batch_size, 128, height, width)
            VAEResidualBlock(256, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 256, height/8, width/8)
            nn.GroupNorm(32, 128),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 256, height/8, width/8)
            nn.SiLU(),

            # (batch_size, 128, height, width) -> (batch_size, 3, height, width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, x:torch.tensor) -> torch.tensor:
        """
        Implementing the forward pass of the decoder module.

        Args:
            x: input tensor of shape (batch_size, 4, height/8, width/8)

        returns:
            Output tensor of shape (batch_size, 3, height, width)
        """
        x /= 0.18215

        for module in self:
            x = module(x)

        return x
