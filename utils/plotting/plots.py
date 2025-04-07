import scipy
import torch
from ipywidgets import interact, FloatSlider
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
from .formatting import make_fig_pretty


__all__ = ["plot_loss", "plot_distribution"]


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

def visualize_neuron(weight1, weight2, bias):
    # Create a neuron with user-specified parameters
    weights = torch.tensor([weight1, weight2])
    bias = torch.tensor([bias])
    
    # Generate a grid of input values
    x1 = torch.linspace(-5, 5, 100)
    x2 = torch.linspace(-5, 5, 100)
    X1, X2 = torch.meshgrid(x1, x2)
    
    # Compute the neuron output for each input pair
    Z = torch.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = torch.tensor([X1[i,j], X2[i,j]])
            # Apply the neuron's linear transformation and activation
            z = x @ weights + bias
            Z[i,j] = torch.sigmoid(z)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X1.numpy(), X2.numpy(), Z.numpy(), 20, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    # plt.title(f'Neuron Output with Weights=[{weight1:.2f}, {weight2:.2f}], Bias={bias:.2f}')
    plt.grid(True)
    plt.show()

# Create interactive sliders
interact(
    visualize_neuron,
    weight1=FloatSlider(min=-2, max=2, step=0.1, value=1.0),
    weight2=FloatSlider(min=-2, max=2, step=0.1, value=-1.0),
    bias=FloatSlider(min=-3, max=3, step=0.1, value=0.0)
)