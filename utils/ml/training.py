import torch

def train_model(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                num_epochs: int = 10,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                early_stopping: bool = False,
                patience: int = 5,
                save_path: str = None) -> torch.nn.Module:
    """"
    "Train a PyTorch model with optional early stopping and model saving."
    ""
    ""
    "   Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        num_epochs (int, optional): Number of epochs to train. Defaults to 10.
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
        patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 5.
        save_path (str, optional): Path to save the best model. Defaults to None.
    Returns:
        torch.nn.Module: The trained model.
    """
    model.to(device)
    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    best_model = model
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
    else:
        print(f'Finished training for {num_epochs} epochs')
    if save_path and best_model:
        model.load_state_dict(torch.load(save_path))
        print(f'Loaded best model from {save_path}')
    return model
