"""
    Building the encoder module of variational encoder and decoder architecture.
"""

import torch
from torch import nn
import torch.nn.functional as F

from decoder import VAEAttentionBlock, VAEResidualBlock

class VariationalEncoder(nn.Sequential):
    """
    To learn the latent space parameters mu and sigma.
    """
    def __init__(self):
        super().__init__(
            # (batch_size, channel, height, width) -> (batch_size, 128, height, width)
            # using the formula height = (height + 2 * padding - kernel_size) / stride + 1
            nn.Conv2d(3, 128, kernel_size=3, padding=1, stride=1),

            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAEResidualBlock(128, 128),

            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAEResidualBlock(128, 128),

            # (batch_size, channel, height, width) -> (batch_size, 128, height/2, width/2)
            # An assymmetrical padding will be added to keep the size consistent.
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=2),

            # (batch_size, 128, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAEResidualBlock(128, 256),

            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAEResidualBlock(256, 256),

            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/4, width/4)
            # An assymmetrical padding will be added to keep the size consistent.
            nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=2),

            # (batch_size, 256, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAEResidualBlock(256, 512),

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAEResidualBlock(512, 512),

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/8, width/8)
            # An assymmetrical padding will be added to keep the size consistent.
            nn.Conv2d(512, 512, kernel_size=3, padding=0, stride=2),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAEResidualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 256, height/8, width/8)
            VAEResidualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 256, height/8, width/8)
            VAEResidualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 256, height/8, width/8)
            VAEAttentionBlock(512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 256, height/8, width/8)
            VAEResidualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 256, height/8, width/8)
            nn.GroupNorm(32, 512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 256, height/8, width/8)
            nn.SiLU(),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1, stride=1),

            # (batch_size, 8, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0, stride=1)
        )

    def enocoder_forward(self, x: torch.tensor, noise: torch.tensor) -> torch.tensor:
        """
        Forward pass of the encoder.

        Args:
            x: input image of the shape (batch_size, 3, height, width)
            noise: random noise of shape (batch_size, output_channel, output_height, output_width)
                    The distribution of noise if pure gaussian N(0, 1).

        Returns:
            Mean and variance of the latent distribution.
        """
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # Apply assymetrical padding to (left, right, top, bottom)
                x = F.pad(0, 1, 0, 1)
            x = module(x)

        # x.shape = (batch_size, 8, height/8, width/8)
        # Get two tensors of shape (batch_size, 4, height/8, width/8) from x
        mean, log_var = torch.chunk(x, 2, dim=1)

        log_var = torch.clamp(log_var, -30, 20)
        var = log_var.exp()
        std = var.sqrt()

        # noise(Z) \in N(0, 1) and to sample from N(mean,std) we have N(mean,std) = mean + std*Z
        x = mean + std*noise
        x *= 0.18215
        return x