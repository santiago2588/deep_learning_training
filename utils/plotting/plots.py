import scipy
import torch
from ipywidgets import interact, FloatSlider
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
from .formatting import make_fig_pretty


__all__ = ["plot_loss", "plot_distribution", "visualize_network"]


def plot_loss(
    train_loss: List[float],
    val_loss: Optional[List[float]] = None,
    title: str = "Training Progress",
    figsize: tuple = (10, 6),
    **kwargs: Optional[dict]
) -> None:
    """
    Plot training and validation loss curves.

    Args:
        train_loss: List of training loss values
        val_loss: Optional list of validation loss values
        title: Plot title
        figsize: Figure size as (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_loss) + 1)

    # Plot training loss
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')

    # Plot validation loss if provided
    if val_loss is not None:
        ax.plot(epochs, val_loss, 'r--', linewidth=2, label='Validation Loss')

    make_fig_pretty(
        ax=ax,
        xlabel="Epoch",
        ylabel="Loss",
        title=title,
        legend=True,
        legd_loc='upper right',
        grid=True
    )

    plt.tight_layout()
    return fig, ax


def plot_distribution(
        data: List[float],
        ax: plt.axes,
        title: str = "Distribution Plot",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        bins: int = 30,
        kdensity: bool = False,
        make_pretty: bool = True,
        **kwargs: Optional[dict]) -> None:

    sharex = kwargs.get('sharex', False)
    sharey = kwargs.get('sharey', False)
    kernel = scipy.stats.gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 1000)
    kde = kernel(x_range)

    ax.hist(data, bins=bins, density=True, alpha=0.5, color='skyblue')

    if kdensity:
        ax.plot(x_range, kde, color='red', linewidth=2, label='KDE')

    if make_pretty:
        make_fig_pretty(
            ax=ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=True,
            legd_loc='upper right',
            grid=False,
            xtick_fsize=10,
            ytick_fsize=10,
            xlabel_fsize=10,
            ylabel_fsize=10,
            title_fsize=10,
            sharex=sharex,
            sharey=sharey
        )
        return
    else:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

# Helper function to visualize the network
def visualize_network(model: torch.nn.Module) -> None:
    """
    Visualize the architecture of a neural network.
    
    Args:
        model: PyTorch neural network model
    """
    # Extract sequential structure of the model
    model_layers = []
    activation_functions = []
    
    # Collect all modules in sequential order
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            model_layers.append({
                'type': 'linear',
                'in_features': module.in_features,
                'out_features': module.out_features,
                'name': name
            })
        elif any(isinstance(module, act_type) for act_type in [
            torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Sigmoid, 
            torch.nn.Tanh, torch.nn.Softmax, torch.nn.GELU,
            torch.nn.ELU, torch.nn.SELU
        ]):
            # Map activation function to readable name
            act_name = type(module).__name__.replace("ReLU", "ReLU").replace("LeakyReLU", "Leaky ReLU")
            activation_functions.append(act_name)
    
    # Extract layer sizes from the model architecture
    layer_sizes = []
    
    # First layer's input size
    if model_layers:
        layer_sizes.append(model_layers[0]['in_features'])
        
    # All layers' output sizes
    for layer in model_layers:
        layer_sizes.append(layer['out_features'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set limits and remove axes
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Load font
    from utils.plotting.fonts import load_font
    custom_font = load_font()
    
    # Layer positions - distribute evenly
    layer_positions = np.linspace(10, 90, len(layer_sizes))
    
    # Draw nodes for each layer
    for i, (layer_pos, layer_size) in enumerate(zip(layer_positions, layer_sizes)):
        # Calculate node positions - distribute evenly vertically
        max_nodes_to_display = min(layer_size, 15)  # Limit nodes for large layers
        if layer_size > max_nodes_to_display:
            node_positions = np.linspace(80, 20, max_nodes_to_display)
            # Add text to indicate there are more nodes
            ax.text(layer_pos, 10, f"(+{layer_size - max_nodes_to_display} more)", 
                    ha='center', fontproperties=custom_font, fontsize=8)
        else:
            node_positions = np.linspace(80, 20, layer_size)
        
        # Draw nodes
        for node_pos in node_positions:
            circle = plt.Circle((layer_pos, node_pos), 3, color='blue', fill=True)
            ax.add_patch(circle)
        
        # Add layer label
        if i == 0:
            ax.text(layer_pos, 90, f'Input Layer\n({layer_size})', ha='center', fontproperties=custom_font)
        elif i == len(layer_sizes) - 1:
            ax.text(layer_pos, 90, f'Output Layer\n({layer_size})', ha='center', fontproperties=custom_font)
        else:
            ax.text(layer_pos, 90, f'Hidden Layer {i}\n({layer_size})', ha='center', fontproperties=custom_font)
        
        # Draw connections to next layer
        if i < len(layer_sizes) - 1:
            next_layer_pos = layer_positions[i + 1]
            next_layer_size = layer_sizes[i + 1]
            
            # Handle large layers by limiting connections
            next_max_nodes = min(next_layer_size, 15)
            next_node_positions = np.linspace(80, 20, next_max_nodes) if next_layer_size > 0 else []
            
            # Draw connections (limit for large layers to avoid clutter)
            max_connections = min(len(node_positions), 15)
            for idx, node_pos in enumerate(node_positions[:max_connections]):
                for next_idx, next_node_pos in enumerate(next_node_positions[:max_connections]):
                    # Skip some connections for better visualization if there are many nodes
                    if max_connections > 10 and (idx % 2 != 0 or next_idx % 2 != 0):
                        continue
                    alpha = 0.1
                    ax.plot([layer_pos, next_layer_pos], [node_pos, next_node_pos], 'k-', alpha=alpha)
            
            # Add activation function label if available
            if i < len(activation_functions):
                midpoint = (layer_pos + next_layer_pos) / 2
                ax.text(midpoint, 5, f'{activation_functions[i]}', 
                        ha='center', fontproperties=custom_font, color='red', fontsize=9)
    
    plt.title('Neural Network Architecture', fontsize=16, fontproperties=custom_font)
    plt.tight_layout()
    
    return fig, ax
