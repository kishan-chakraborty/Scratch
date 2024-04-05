"""
Implementing encoder module of auto-encoder architecture.
"""

from torch import nn

class AutoEncoder(nn.Module):
    """
    Building the autoencoder architecture from scratch.

    """

    def __init__(self, input_dim:int=784) -> None:
        super().__init__()

        self.hidden_dim = 8
        # The encoded latent representation of input image.
        self.encoded = None
        # Reconstruction of the input image obtained by decoding latent representation.
        self.decoded = None

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Implementing the forward pass of the autoencoder to reconstruct the input training data.

        Args:
            x: input training data of the shape (batch_size, n_features)

        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
