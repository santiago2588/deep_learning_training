from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

__all__ = ['r2_score', 'compute_accuracy', 'compute_classification_report']


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate R^2 score (coefficient of determination) for regression models.
    
    The R^2 score measures how well the predictions approximate the true values.
    A score of 1.0 indicates perfect predictions, while 0.0 indicates a model 
    that performs no better than always predicting the mean of the data.
    
    Args:
        y_true (torch.Tensor): True target values
        y_pred (torch.Tensor): Predicted target values
    
    Returns:
        float: R^2 score between 0.0 and 1.0 (higher is better)
    """
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2.item()


def _get_predictions(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get predictions from model for a given dataloader.
    
    This function runs a model on all samples in a dataloader and collects
    the true labels and model predictions for evaluation.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): Dataloader with evaluation data
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Lists of true labels and predictions
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
    Compute classification accuracy between true and predicted values.
    
    This function evaluates a classification model on a dataset and returns
    the proportion of samples that were correctly classified.
    
    Args:
        model (torch.nn.Module): Classification model to evaluate
        dataloader (torch.utils.data.DataLoader): Dataloader with evaluation data
    
    Returns:
        float: Accuracy score between 0.0 and 1.0 (higher is better)
    """
    y_true, y_pred = _get_predictions(model, dataloader)
    acc = accuracy_score(y_true, y_pred)
    return acc


def compute_classification_report(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, class_names: list[str] = None) -> str:
    """
    Compute a comprehensive classification report with precision, recall, and F1 metrics.
    
    This function generates a detailed classification report including precision,
    recall, F1-score, and support for each class, as well as macro and weighted averages.
    
    Args:
        model (torch.nn.Module): Classification model to evaluate
        dataloader (torch.utils.data.DataLoader): Dataloader with evaluation data
        class_names (list[str], optional): Names of the classes for reporting
    
    Returns:
        str: Text classification report showing precision, recall, F1 score, and support for each class
    """
    y_true, y_pred = _get_predictions(model, dataloader)
    report = classification_report(y_true, 
                                   y_pred,
                                   target_names=class_names)
    return report
