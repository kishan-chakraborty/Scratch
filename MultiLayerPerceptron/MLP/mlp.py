"""
Building a multi-layer perceptron from scrstch using torch.array.
No use of inbuilt pytorch neural network module.
"""
import torch
from utils import tanh, dtanh, cross_entropy_loss

class Linear:
    """
    Building linear layer of a feed-forward neural network.
    """
    def __init__(self, n_input:int, n_output:int, activation:str = 'tanh') -> None:
        """
        Linear hidden layer
        
        Args:
            n_input (int): Dimension of the input data to the layer.
            n_output (int): Dimension of the output data from the layer.
            activation (str, optional): Activation function. Defaults to 'tanh'.

        return:
            None
        """
        self.w = torch.randn(n_input, n_output)
        self.b = torch.ones(n_output)
        self.activation = activation
        self.a = None
        self.h = None
        self.da = None
        self.dh = None
        self.dw = None
        self.db = None

    def __call__(self, x):
        self.a = x @ self.w + self.b
        self.h = tanh(self.a)
        return self.h


class MLP():
    """
    Implementing feed-forward neural network.
    """
    def __init__(self, layers: list) -> None:
        """
        Build a neural network by passing list of layers.

        Args:
            list of hidden and output layers. 
        """
        self.layers = layers    # list of hidden and output layers
        self.x = None           # Training predictor variables
        self.y = None           # Training response variables
        self.y_pred = None      # Training prediction
        self.n_train = None     # Number of training samples
        self.n_feat = None      # No. of predictor features
        self.n_classes = None   # No. of classes
        self.losses = []        # List of losses during training

    def fit(self, x: torch.tensor, y: torch.tensor, epochs:int=10, lr:torch.float=0.01) -> None:
        """
        Training the neural netowrk on the given training data.

        Args: 
            x: input training data of the shape (batch_size, n_features)
            y: output training data of the shape (batch_size, k_classes)
        """
        self.x = x
        self.y = y
        self.n_train, self.n_feat = x.shape
        self.n_classes = y.shape[1]
        for _ in range(epochs):
            self.y_pred = self.forward(x)
            loss = cross_entropy_loss(y_true=y, y_pred=self.y_pred)
            self.losses.append(loss)
            print(loss)
            self.backward()

            # Update the parameters
            for layer in self.layers:
                layer.w -= lr * layer.dw
                layer.b -= lr * layer.db

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the neural network.

        Args:
            x: input training data of the shape (batch_size, n_features)

        return:
            Resultant after the forward pass of the neural network of shape (batch_size, k_classes)
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self):
        """
        Implementing the backward pass of the neural network and updating parameters.
        """
        # find the gradient corresponding to the parameters of final layer.
        self._final_layer_grad()
        for i in range (len(self.layers)-2, 0, -1):
            dhi = self.layers[i+1].da @ self.layers[i+1].w.T
            self.layers[i].dh = dhi
            dai = dhi * dtanh(self.layers[i].a)
            self.layers[i].da = dai
            self.layers[i].dw = self.layers[i-1].h.T @ dai
            self.layers[i].db = dai.mean(dim=0)

        dhi = self.layers[1].da @ self.layers[1].w.T
        self.layers[0].dh = dhi
        dai = dhi * dtanh(self.layers[0].a)
        self.layers[0].da = dai
        self.layers[0].dw = self.x.T @ dai
        self.layers[0].db = dai.mean(dim=0)

    def _final_layer_grad(self) -> None:
        """
        Implment th gradient corresponding to the parameters of final layer.
        """
        final_layer = self.layers[-1]
        class_labels = torch.argmax(self.y, dim=1)   # Find the labels for each example.
        grad_yhat = torch.zeros_like(self.y)
        grad_yhat[torch.arange(self.n_train), class_labels] = \
                -1/(self.y_pred[torch.arange(self.n_train), class_labels])
        final_layer.dh = grad_yhat
        final_layer.da = self.y_pred - self.y
        final_layer.dw = self.layers[-2].h.T @ final_layer.da
        final_layer.db = final_layer.da.mean(dim=0)


if __name__ == '__main__':
    # Input data. 2 example with 3 features.
    x = torch.tensor([[1, 0, 1],
                      [2, 1, 0],
                      [1, 0, 1],
                      [2, 1, 0]], dtype=torch.float)
    y = torch.tensor([[1, 0],
                      [0, 1],
                      [1, 0],
                      [0, 1]], dtype=torch.float)
    Linear1 = Linear(3, 2)
    Linear2 = Linear(2, 3)
    Linear3 = Linear(3, 2)
    model = [Linear1, Linear2, Linear3]
    mlp = MLP(model)
    mlp.fit(x, y)