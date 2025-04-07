import scipy
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
from .formatting import make_fig_pretty


__all__ = ["plot_loss", "plot_distribution"]

def plot_loss(
    train_loss: List[float],
    val_loss: Optional[List[float]] = None,
    title: str = "Training Progress",
    figsize: tuple = (10, 6)
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
    make_pretty: bool = True,):

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
        )
        return
    else:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax