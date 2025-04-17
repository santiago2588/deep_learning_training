from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

__all__ = ['r2_score', 'compute_accuracy', 'compute_classification_report']


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


def _get_predictions(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get predictions from model for a given dataloader.

    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): Dataloader with evaluation data

    Returns:
        tuple[torch.Tensor, torch.Tensor]: True and predicted labels
    """
    device = next(model.parameters()).device
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return y_true, y_pred


def compute_accuracy(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> float:
    """
    Compute accuracy between true and predicted values.

    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): Dataloader with evaluation data

    Returns:
        float: Accuracy score.
    """
    y_true, y_pred = _get_predictions(model, dataloader)
    acc = accuracy_score(y_true, y_pred)
    return acc


def compute_classification_report(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, class_names: list[str] = None) -> str:
    """
    Compute classification report between true and predicted values.

    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): Dataloader with evaluation data

    Returns:
        str: Classification report.
    """
    y_true, y_pred = _get_predictions(model, dataloader)
    report = classification_report(y_true, 
                                   y_pred,
                                   target_names=class_names)
    return report
