import torch
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import FloatSlider, interact, interactive, Layout,interactive_output
from typing import Callable, Optional, Dict, Any, Union, Tuple

from .plots import make_fig_pretty

__all__ = ["create_interactive_neuron_visualizer"]

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
