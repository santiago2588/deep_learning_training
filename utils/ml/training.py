import torch
from tqdm.auto import tqdm
from typing import Dict, Any
from ..plotting.plots import plot_loss

__all__ = ['train_model']

def train_model(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    **kwargs: Dict[str, Any]
) -> torch.nn.Module:
    """
    Train a PyTorch model with flexible validation options.

    Args:
        model: The model to train
        criterion: Loss function
        optimiser: Optimizer
        **kwargs: Additional arguments
            Training data (one of):
            - train_loader: DataLoader for training data
            OR
            - train_inputs: Training input tensor
            - train_targets: Training target tensor
            
            Optional validation data:
            - val_loader: DataLoader for validation data
            OR
            - val_inputs: Validation input tensor
            - val_targets: Validation target tensor
            
            Other optional parameters:
            - batch_size: Batch size for tensor inputs (default: 32)
            - num_epochs (int): Number of epochs (default: 10)
            - device (str): Device to use (default: 'cuda' if available else 'cpu')
            - early_stopping (bool): Whether to use early stopping (default: False)
            - patience (int): Epochs with no improvement before stopping (default: 5)
            - tolerance (float): Minimum change in validation loss to qualify as improvement (default: 1e-4)
            - save_path (str): Path to save best model (default: None)
            - verbose (bool): Whether to show progress bars (default: True)

    Returns:
        torch.nn.Module: The trained model
    """
    # Extract configuration from kwargs with defaults
    config = {
        'num_epochs': kwargs.get('num_epochs', 10),
        'device': kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        'early_stopping': kwargs.get('early_stopping', False),
        'patience': kwargs.get('patience', 5),
        'tolerance': kwargs.get('tolerance', 1e-4),
        'save_path': kwargs.get('save_path', None),
        'verbose': kwargs.get('verbose', True),
        'batch_size': kwargs.get('batch_size', 32),
        'plot_loss': kwargs.get('plot_loss', False),
        'return_best_model': kwargs.get('return_best_model', False)
    }

    if config['save_path']:
        try:
            config['save_path'].parent.mkdir(parents=True, exist_ok=True)
        except AttributeError:
            raise ValueError("save_path must be a pathlib.Path object")


    # Determine data input mode
    using_train_loader = 'train_loader' in kwargs
    using_train_tensors = all(k in kwargs for k in ['train_inputs', 'train_targets'])
    using_val_loader = 'val_loader' in kwargs
    using_val_tensors = all(k in kwargs for k in ['val_inputs', 'val_targets'])

    if not (using_train_loader or using_train_tensors):
        raise ValueError("Must provide either train_loader or (train_inputs, train_targets)")

    # Set up training data
    if using_train_loader:
        train_loader = kwargs['train_loader']
    else:
        train_dataset = torch.utils.data.TensorDataset(
            kwargs['train_inputs'].to(config['device']),
            kwargs['train_targets'].to(config['device'])
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=config['batch_size'],
            shuffle=True
        )

    # Set up validation data if provided
    has_validation = using_val_loader or using_val_tensors
    if has_validation:
        if using_val_loader:
            val_loader = kwargs['val_loader']
        else:
            val_dataset = torch.utils.data.TensorDataset(
                kwargs['val_inputs'].to(config['device']),
                kwargs['val_targets'].to(config['device'])
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config['batch_size']
            )

    # Disable early stopping if no validation set
    if not has_validation and config['early_stopping']:
        if config['verbose']:
            print("Early stopping disabled - no validation set provided")
        config['early_stopping'] = False

    model.to(config['device'])
    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    epoch_pbar = tqdm(range(config['num_epochs']), desc=f'{model.__class__.__name__} Training', position=0, leave=True) if config['verbose'] else range(config['num_epochs'])
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_iter = train_loader
        n_batches = len(train_loader)

        if config['verbose']:
            train_iter = tqdm(train_loader, desc=f"Training Batch 0/{n_batches}", position=1, leave=False)
            train_iter.set_postfix({'batch_loss [Training]': '0.0000'})

        for batch in train_iter:
            inputs, targets = batch
            inputs = inputs.to(config['device'])
            targets = targets.to(config['device'])

            optimiser.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimiser.step()

            train_loss += loss.item()
            if config['verbose']:
                train_iter.set_postfix({'batch_loss [Training]': f'{loss.item():.4f}'})
                train_iter.set_description(f"Training Batch {train_iter.n}/{n_batches}")

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_iter.close()

        # Validation phase (if validation data provided)
        if has_validation:
            model.eval()
            val_loss = 0.0
            val_iter = val_loader
            n_batches = len(val_loader)
            
            if config['verbose']:
                val_iter = tqdm(val_loader, desc=f"Validation Batch 0/{n_batches}", position=1, leave=False)
                val_iter.set_postfix({'batch_loss [Validation]': '0.0000'})

            with torch.no_grad():
                for batch in val_iter:
                    inputs, targets = batch
                    inputs = inputs.to(config['device'])
                    targets = targets.to(config['device'])

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    if config['verbose']:
                        val_iter.set_postfix({'batch_loss [Validation]': f'{loss.item():.4f}'})
                        val_iter.set_description(f"Validation Batch {val_iter.n}/{n_batches}")

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            val_iter.close()

            # Track best validation loss regardless of early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if config['save_path']:
                    torch.save(model.state_dict(), config['save_path'])
                    best_model = model


            # Update epoch progress bar with both losses
            if config['verbose']:
                epoch_pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'best_val_loss': f'{best_val_loss:.4f}'
                })

            # Early stopping logic
            if config['early_stopping']:
                if (val_loss - config['tolerance'] < best_val_loss) or (val_loss < train_loss):
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= config['patience']:
                        if config['verbose']:
                            epoch_pbar.write(f'Early stopping at epoch {epoch+1}')
                        break
        else:
            # Update epoch progress bar with only training loss
            if config['verbose']:
                epoch_pbar.set_postfix({'train_loss': f'{train_loss:.4f}'})
    
    epoch_pbar.close()
    
    if config['save_path'] and best_model and has_validation:
        if config['return_best_model']:
            model.load_state_dict(torch.load(config['save_path']))
            if config['verbose']:
                print(f'Loaded best model from {config["save_path"]}')
        else:
            if config['verbose']:
                print(f'Best model saved at {config["save_path"]}')
    
    if config['plot_loss']:
        plot_loss(train_losses, val_losses)

    return model