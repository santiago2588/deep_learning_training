import torch

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate R^2 score.
    
    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.
        
    Returns:
        float: R^2 score.
    """
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2.item()

def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate accuracy.
    
    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.
        
    Returns:
        float: Accuracy score.
    """
    correct = (y_true == y_pred).sum().item()
    total = y_true.size(0)
    return correct / total if total > 0 else 0.0