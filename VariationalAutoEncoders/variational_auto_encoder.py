import torch
from torch import nn
from torch import optim

from encoder import AutoEncoder

class VariationalAutoEncoder(AutoEncoder):
    """
    Implementing variational auto encoder mocule.
    """

    def __init__(self):
        super().__init__()
        self.mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.log_var = nn.Linear(self.hidden_dim, self.hidden_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Implementing the reparameterization trick.

        Args:
            mu: mean of the latent distribution which is to be estimated.
            log_var: log variance of the latent distribution which is to be estimated.

        return: The gaussian distribution of the latent space.
        """
        # standard deviation.
        std = torch.exp(0.5*log_var)
        # Generate random gaussian noise.
        eps = torch.randn_like(std)
        out = mu + std * eps

        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                 torch.Tensor, torch.Tensor]:
        """
        Implementing the forward pass of the VAE.

        Args:
            x: Imput data of shape [batch_size, feature_dim]
        """
        encoded = self.encoder(x)
        # We try to model the mean and variance of the latent distribution using input data
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)

        decoded = self.decoder(z)

        return encoded, decoded, mu, log_var

    def sample(self, device, num_sample: int) -> torch.Tensor:
        """
        Generate sample from the latent space.
        """
        with torch.no_grad():
        # Generate random gaussian sample
            z = torch.randn(num_sample, self.hidden_dim).to(device)
            sample = self.decoder(z)

        return sample

def train_variational_autoencoder(device, model: VariationalAutoEncoder, 
                                  x_train: torch.Tensor,
                                  n_epochs:int=50, batch_size:int=64,
                                  learning_rate:float=0.1):
    """
    Function to train the auto encoder.

    Args:
        device: CPU/ GPU.
        model: Variational auto encoder model.
        x_train: Training data

    return:
        Trained model.
    """
    # Convert the training data to PyTorch tensors
    x_train = x_train.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define the loss function
    criterion = nn.MSELoss(reduction="sum")

    model.to(device)

    train_loader = torch.utils.data.DataLoader(
        x_train, batch_size=batch_size, shuffle=True
    )

    for epoch in range(n_epochs):
        total_loss = 0.0
        for _, data in enumerate(train_loader):
            # Get a batch of training data and move it to the device
            data = data.to(device)

            # Forward pass
            _, decoded, mu, log_var = model(data)

            # Compute the loss and perform backpropagation
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = criterion(decoded, data) + 3 * KLD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the running loss
            total_loss += loss.item() * data.size(0)

    # Print the epoch loss
    epoch_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{n_epochs}: loss={epoch_loss:.4f}")

    # Return the trained model
    return model
