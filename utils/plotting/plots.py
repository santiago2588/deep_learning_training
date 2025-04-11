import scipy
import torch
from ipywidgets import interact, FloatSlider
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from .formatting import make_fig_pretty
import networkx as nx
import matplotlib.patches as mpatches
from .fonts import load_font
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D


__all__ = ["plot_loss", "plot_distribution",
           "visualize_network_nx"]


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


def visualize_network_nx(model: torch.nn.Module, figsize: tuple = (8, 4)) -> Tuple[plt.Figure, plt.Axes, nx.DiGraph, Dict[str, Any]]:
    """
    Visualize a neural network architecture using networkx.

    Args:
        model: PyTorch neural network model
        figsize: Figure size as (width, height)

    Returns:
        fig, ax, G, pos: Matplotlib figure, axes, networkx graph, and node positions
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Extract model layers and their dimensions
    layers = []
    layer_names = []
    activations = []

    # Process all modules in the model
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            layers.append({
                'type': 'linear',
                'in_features': module.in_features,
                'out_features': module.out_features,
                'name': name
            })
            layer_names.append(name)
        elif any(isinstance(module, act_type) for act_type in [
            torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Sigmoid,
            torch.nn.Tanh, torch.nn.Softmax, torch.nn.GELU
        ]):
            activations.append(type(module).__name__)

    # For simple models or manually created ones, extract parameters
    if not layers and hasattr(model, 'weights'):
        try:
            in_features = model.weights.shape[0] if len(
                model.weights.shape) == 1 else model.weights.shape[1]
            out_features = 1
            layers.append({
                'type': 'custom',
                'in_features': in_features,
                'out_features': out_features,
                'name': 'custom_layer'
            })
            layer_names.append('custom_layer')
            if hasattr(model, 'activation'):
                activations.append(model.activation.__name__ if hasattr(
                    model.activation, '__name__') else 'Custom')
        except (AttributeError, IndexError):
            # Fall back to a simple representation if structure can't be determined
            in_features = 1
            out_features = 1

    # If still no layers found, create a placeholder
    if not layers:
        print("Warning: Could not determine network structure automatically")
        return None, None, None, None

    # Simplify the network representation
    # Create basic nodes: inputs, hidden layer, and output
    input_nodes = [f"x{i+1}" for i in range(layers[0]['in_features'])]
    hidden_nodes = ["Neuron"] if layers[0]['out_features'] == 1 else [f"h{i+1}" for i in range(layers[0]['out_features'])]
    output_nodes = ["y"] if len(layers) == 1 or layers[-1]['out_features'] == 1 else [f"y{i+1}" for i in range(layers[-1]['out_features'])]
    
    # Add all nodes to the graph
    for node in input_nodes + hidden_nodes + output_nodes:
        G.add_node(node)
    
    # Add edges from inputs to hidden layer
    for input_node in input_nodes:
        for hidden_node in hidden_nodes:
            G.add_edge(input_node, hidden_node)
    
    # If we have a deeper network, add edges from hidden to output
    if len(layers) > 1:
        for hidden_node in hidden_nodes:
            for output_node in output_nodes:
                G.add_edge(hidden_node, output_node)
    else:
        # Add direct edges from hidden to output for single layer networks
        for hidden_node in hidden_nodes:
            for output_node in output_nodes:
                G.add_edge(hidden_node, output_node)
    
    # Create a simple, clean layout
    pos = {}
    
    # Position input nodes vertically on the left
    input_spacing = 2 / max(len(input_nodes), 1)
    for i, node in enumerate(input_nodes):
        pos[node] = (-1, (i * input_spacing) - 1 + input_spacing/2)
    
    # Position hidden nodes in the middle
    hidden_spacing = 2 / max(len(hidden_nodes), 1)
    for i, node in enumerate(hidden_nodes):
        pos[node] = (0, (i * hidden_spacing) - 1 + hidden_spacing/2)
    
    # Position output nodes on the right
    output_spacing = 2 / max(len(output_nodes), 1)
    for i, node in enumerate(output_nodes):
        pos[node] = (1, (i * output_spacing) - 1 + output_spacing/2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Load font for consistent styling - use the same font for everything
    fm = load_font()
    
    # Define pleasing colors
    node_color = '#AED6F1'  # Light blue
    edge_color = '#2C3E50'  # Dark blue/gray
    activation_color = '#E74C3C'  # Red for activation functions
    
    # Draw nodes first (so edges appear on top)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=2000,
        node_color=node_color,
        edgecolors='black',
        linewidths=1.5,
        ax=ax
    )
    
    # Identify activation edges - always the connections to output layer
    act_edges = []
    regular_edges = []
    
    for u, v in G.edges():
        if v in output_nodes:
            act_edges.append((u, v))
        else:
            regular_edges.append((u, v))
    
    # Draw regular edges with straight lines for better arrow visibility
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=regular_edges,
        arrowsize=20,
        width=1.5,
        alpha=0.7,
        edge_color=edge_color,
        connectionstyle='arc3,rad=0.0',  # Straight lines for clarity
        arrows=True,  # Ensure arrows are shown
        arrowstyle='-|>',  # Clear arrow style
        ax=ax
    )
    
    # Draw activation function edges with different style and color
    if activations and act_edges:
        # Get activation name for display in legend
        act_name = ", ".join(set(activations))
        
        # Draw activation edges with distinctive style
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=act_edges,
            arrowsize=25,
            width=3.0,
            alpha=1.0,
            edge_color=activation_color,
            connectionstyle='arc3,rad=0.0',  # Straight lines for better arrows
            style='solid',
            arrows=True,
            arrowstyle='-|>',  # More visible arrow style
            min_source_margin=15,  # Give space for the arrow to be visible
            min_target_margin=15,
            ax=ax
        )
        
        # We no longer add the dot on the activation function lines
    
    # Draw node labels with consistent font
    label_font = fm.copy()
    label_font.set_weight('bold')
    
    # Custom draw labels with consistent font
    for node, (x, y) in pos.items():
        ax.text(x, y, node, 
                fontproperties=label_font,
                fontsize=10,
                horizontalalignment='center',
                verticalalignment='center',
                color='black',
                transform=ax.transData,
                zorder=12)
    
    # Add a title with network size information
    n_input = len(input_nodes)
    n_hidden = len(hidden_nodes)
    n_output = len(output_nodes)
    title = f"Neural Network ({n_input}-{n_hidden}-{n_output})"
    if activations:
        title += "\nActivations: " + ", ".join(set(activations))
    
    # Add legend for activation function if present
    legend_elements = [
        mpatches.Patch(color=node_color, label='Network Nodes'),
        Line2D([0], [0], color=edge_color, lw=1.5, label='Connections')
    ]
    
    if activations:
        legend_elements.append(
            Line2D([0], [0], color=activation_color, lw=3.0, 
                  label=f'{act_name} Activation')  # Removed marker
        )
    
    # Create the legend with consistent font in the top-left position
    legend = ax.legend(
        handles=legend_elements, 
        loc='upper left',  # Changed to upper left as requested
        prop=fm,
        framealpha=0.8  # Better visibility of legend
    )
    
    # Make figure pretty with consistent font throughout
    make_fig_pretty(
        ax=ax,
        title=title,
        grid=False
    )
    
    # Make sure there's enough padding around the graph
    plt.tight_layout(pad=1.2)
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    return fig, ax, G, pos
