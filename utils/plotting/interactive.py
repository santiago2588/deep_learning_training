import torch
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import FloatSlider, interact, interactive, Layout,interactive_output
from typing import Callable, Optional, Dict, Any, Union, Tuple

from .plots import make_fig_pretty

__all__ = ["create_interactive_neuron_visualizer", "se04_visualize_transformations", "wake_cylinder"]

def wake_cylinder(X_star:np.array, u_star:np.array, p_star:np.array, time:np.array):
    
    # Reshape grid
    x_unique = np.unique(X_star[:,0])
    y_unique = np.unique(X_star[:,1])
    Nx, Ny = len(x_unique), len(y_unique)

    assert Nx*Ny == X_star.shape[0], "Grid reshaping failed"

    x = X_star[:,0].reshape(Ny, Nx)
    y = X_star[:,1].reshape(Ny, Nx)

    fig, ax = plt.subplots(figsize=(8, 6))

    cyl_rad = 0.1
    cyl_center = (0.0, 0.0)

    def refresh_plot(ix):
        ax.clear()

        u = u_star[:,0,ix].reshape(Ny, Nx)
        v = u_star[:,1,ix].reshape(Ny, Nx)
        p = p_star[:,ix].reshape(Ny, Nx)

        ax.contourf(x, y, p, levels=20, cmap='viridis')
        ax.streamplot(x, y, u, v, color='white', density=1.5)

        circle = plt.Circle(cyl_center, cyl_rad, color='gray', alpha=0.6)
        ax.add_patch(circle)

        make_fig_pretty(
            ax=ax,
            title=f"Time: {time[ix,0]:.2f} s",
            xlabel="X",
            ylabel="Y",
            xlim=(-10, 10),
            ylim=(-3, 3),
            legend=False,
            grid=False
        )

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()

    slider = widgets.IntSlider(0,0, u_star.shape[2]-1, step=1, description='Time Step')
    widgets.interact(refresh_plot, ix=slider)
    


def _update_plot(
    w: float, 
    b: float, 
    X: torch.Tensor, 
    y: torch.Tensor, 
    X_mean: float, 
    X_std: float, 
    challenger_temp: float, 
    neuron_class: type, 
    loss_function: Callable,
    out: widgets.Output
) -> None:
    """
    Internal function to update the visualization plot based on slider values.
    
    Args:
        w: Weight parameter value
        b: Bias parameter value
        X: Feature tensor
        y: Target tensor
        X_mean: Mean of the features (for normalization)
        X_std: Standard deviation of the features (for normalization)
        challenger_temp: The Challenger launch temperature
        neuron_class: The class used to create the neuron model
        loss_function: Function to compute the loss
        out: Output widget to display the plot
    """
    with out:
        out.clear_output()
        
        # Create a neuron with the specified parameters
        custom_model = neuron_class(n_features=1)
        with torch.no_grad():
            # Set the weights and bias to the specified values
            custom_model.weights.copy_(torch.tensor([w]))
            custom_model.bias.copy_(torch.tensor(b))
        
        # Calculate predictions
        predictions = torch.zeros_like(y)
        X_normalized = X  # Assuming X is already normalized
        
        for i in range(len(X_normalized)):
            predictions[i] = custom_model(X_normalized[i])
        
        # Calculate loss
        loss = loss_function(predictions, y)
        
        # Convert tensors to numpy for plotting
        temperatures = X.numpy().flatten() * X_std + X_mean  # De-normalize for plotting
        actual = y.numpy().flatten()
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot data points
        ax.scatter(temperatures, actual, color='None', edgecolor='k',
                    s=100, alpha=0.7, label='Actual Data')
        
        # Plot model predictions
        temp_range = np.linspace(min(temperatures) - 2, max(temperatures) + 2, 100)
        temp_range_normalized = (temp_range - X_mean) / X_std
        smooth_preds = np.array([custom_model(torch.tensor([t]).float()).item() for t in temp_range_normalized])
        ax.plot(temp_range, smooth_preds, 'orange', linewidth=3, label='Model Prediction')
        
        # Add reference lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=challenger_temp, color='black', linestyle='--', alpha=0.7)
        ax.text(challenger_temp+0.5, 0.8, f'Challenger Launch: {challenger_temp:.1f}°C', rotation=90)
        
        # Calculate error visually
        for i, (temp, true_val, pred_val) in enumerate(zip(temperatures, actual, predictions.detach().numpy())):
            if abs(true_val - pred_val.item()) > 0.05:  # Only show significant errors
                ax.plot([temp, temp], [true_val, pred_val.item()], 'b-', alpha=0.3)
        
        make_fig_pretty(
            ax=ax,
            title=f"O-Ring Failure Model (Loss: {loss.item():.4f})",
            xlabel="Temperature (°C)",
            ylabel="Probability of Failure",
            legend=True,
            legd_loc='upper right',
            grid=True
        )
        
        plt.tight_layout()
        plt.show()

def _train_model_callback(
    b: widgets.Button, 
    w_slider: widgets.FloatSlider, 
    b_slider: widgets.FloatSlider,
    X: torch.Tensor, 
    y: torch.Tensor, 
    neuron_class: type, 
    loss_function: Callable,
    learning_rate: float = 0.1,
    epochs: int = 100
) -> None:
    """
    Callback function to train the model and update sliders.
    
    Args:
        b: Button widget that triggered the callback
        w_slider: Slider for weight parameter
        b_slider: Slider for bias parameter
        X: Feature tensor
        y: Target tensor
        neuron_class: The class used to create the neuron model
        loss_function: Function to compute the loss
        learning_rate: Learning rate for optimization
        epochs: Number of training epochs
    """
    # Reset model
    new_model = neuron_class(n_features=1)
    
    # Train for specified epochs
    optimizer = torch.optim.SGD([new_model.weights, new_model.bias], lr=learning_rate)
    for _ in range(epochs):
        # Forward pass
        predictions = torch.zeros_like(y)
        for i in range(len(X)):
            predictions[i] = new_model(X[i])
        
        # Compute loss
        loss = loss_function(predictions, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Update sliders to the trained values
    w_slider.value = float(new_model.weights.item())
    b_slider.value = float(new_model.bias.item())

def create_interactive_neuron_visualizer(
    X: torch.Tensor, 
    y: torch.Tensor, 
    X_mean: float, 
    X_std: float, 
    challenger_temp: float,
    neuron_class: type, 
    loss_function: Callable,
    initial_weight: Optional[float] = None,
    initial_bias: Optional[float] = None,
    learning_rate: float = 0.1,
    training_epochs: int = 100
) -> widgets.Widget:
    """
    Create an interactive visualization widget for exploring neuron parameters.
    
    Args:
        X: Feature tensor (normalized)
        y: Target tensor
        X_mean: Mean of the features (before normalization)
        X_std: Standard deviation of the features (before normalization)
        challenger_temp: The Challenger launch temperature
        neuron_class: The class used to create the neuron model
        loss_function: Function to compute the loss
        initial_weight: Initial weight value for the slider (default uses model's weight)
        initial_bias: Initial bias value for the slider (default uses model's bias)
        learning_rate: Learning rate for the training button
        training_epochs: Number of epochs when training button is clicked
        
    Returns:
        A widget containing sliders and a train button for interactive visualization
    """
    # Create a temporary model to get default parameter values if not provided
    if initial_weight is None or initial_bias is None:
        temp_model = neuron_class(n_features=1)
        if initial_weight is None:
            try:
                initial_weight = float(temp_model.weights.item())
            except:
                initial_weight = 0.0
        if initial_bias is None:
            try:
                initial_bias = float(temp_model.bias.item())
            except:
                initial_bias = 0.0
    
    # Create output widget and parameter sliders
    out = widgets.Output()
    w_slider = FloatSlider(value=initial_weight, min=-10.0, max=10.0, step=0.1, description='Weight:')
    b_slider = FloatSlider(value=initial_bias, min=-10.0, max=10.0, step=0.1, description='Bias:')
    train_button = widgets.Button(description="Train Model")
    
    # Wrap update_plot with fixed parameters
    def wrapped_update_plot(w, b):
        _update_plot(w, b, X, y, X_mean, X_std, challenger_temp, neuron_class, loss_function, out)
    
    # Connect callbacks
    train_button.on_click(lambda b: _train_model_callback(
        b, w_slider, b_slider, X, y, neuron_class, loss_function, learning_rate, training_epochs
    ))
    
    # Create the interactive widget
    inter_out = interactive_output(wrapped_update_plot, {'w':w_slider, 'b':b_slider})
    
    # Combine all widgets
    main_ui = widgets.VBox([
        widgets.HBox([w_slider, b_slider, train_button]), 
        out
    ])
    
    # Initial update
    wrapped_update_plot(w_slider.value, b_slider.value)
    
    return main_ui, inter_out

def se04_visualize_transformations(transformed_images):
    """Visualize the transformed images.
    
    Args:
        transformed_images (dict): Dictionary of transformed images
    """
    _, axes = plt.subplots(3, 3, figsize=(8, 8))
    axes = axes.flatten()
    
    for i, (title, img_tensor) in enumerate(transformed_images.items()):
        if title == 'Normalized':
            # Denormalize for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = img_tensor * std + mean
            
        img_numpy = img_tensor.permute(1, 2, 0).numpy()
        img_numpy = np.clip(img_numpy, 0, 1)
        
        axes[i].imshow(img_numpy)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()