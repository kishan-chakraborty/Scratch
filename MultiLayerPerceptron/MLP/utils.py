"""
Utility functions for MLP
"""
import torch
def tanh(x:torch.tensor) -> torch.tensor:
    """
    Calculate the tanh function for the input x.

    Args:
        x: input tensor

    Returns:
        applied tanh function on the input tensor x.
    """
    out = torch.tanh(x)
    return out

def dtanh(x:torch.tensor) -> torch.tensor:
    """
    Calculate the differentiation of tanh function for the input x.

    Args:
        x: input tensor

    Returns:
        applied differentiation of tanh function on the input tensor x.
    """
    out = 1 - torch.tanh(x) ** 2
    return out

def cross_entropy_loss(y_true:torch.tensor, y_pred:torch.tensor) -> float:
    """
    Calculate the cross entropy loss for the given prediction and true values.

    Args:
        y_true: true output of shape (batch_size, k_classes) ith row corresponding to the ith example has only one 1 and remaining are 0s.
        y_pred: predicted output of shape (batch_size, k_classes).
    
    Return:
        loss: cross entropy loss.
    """
    log_pred = y_pred.log()
    loss = -(y_true * log_pred).sum(dim=1).sum(dim=0)
    return loss
