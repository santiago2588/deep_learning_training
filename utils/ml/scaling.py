import torch
import tqdm
from PIL import Image
from typing import List, Tuple, Union

__all__ = ['CustomScaler', 'resize_images_in_folder']

class CustomScaler:
    """
    A utility class for scaling and normalizing PyTorch tensors.
    
    This class provides functionality similar to scikit-learn's scalers
    but designed specifically for PyTorch tensors, supporting both standard
    scaling (zero mean, unit variance) and min-max scaling (0-1 range).
    
    Attributes:
        method (str): The scaling method to use ('standard' or 'minmax')
        mean (torch.Tensor): Mean values calculated during fit (standard scaling)
        std (torch.Tensor): Standard deviation values calculated during fit (standard scaling)
        min (torch.Tensor): Minimum values calculated during fit (minmax scaling)
        max (torch.Tensor): Maximum values calculated during fit (minmax scaling)
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize the CustomScaler with the specified scaling method.
        
        Args:
            method (str): The scaling method to use, either 'standard' or 'minmax'
                          'standard': scales to zero mean and unit variance
                          'minmax': scales to the range [0, 1]
        """
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    def fit(self, data:torch.Tensor):
        """
        Calculate scaling parameters from the input data.
        
        Args:
            data (torch.Tensor): Input tensor to calculate scaling parameters from
            
        Returns:
            self: The fitted scaler instance (for method chaining)
            
        Raises:
            ValueError: If an unknown scaling method is specified
        """
        if self.method == 'standard':
            self.mean = torch.mean(data, dim=0)
            self.std = torch.std(data, dim=0)
        elif self.method == 'minmax':
            self.min = torch.min(data, dim=0).values
            self.max = torch.max(data, dim=0).values
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        return self
    
    def transform(self, data:torch.Tensor):
        """
        Apply scaling to the input data using parameters from fit.
        
        Args:
            data (torch.Tensor): Input tensor to scale
            
        Returns:
            torch.Tensor: Scaled tensor
            
        Raises:
            ValueError: If an unknown scaling method is specified
        """
        if self.method == 'standard':
            return (data - self.mean) / (self.std + 1e-8)
        elif self.method == 'minmax':
            return (data - self.min) / (self.max - self.min + 1e-8)
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
    def inverse_transform(self, data:torch.Tensor):
        """
        Revert scaled data back to the original scale.
        
        Args:
            data (torch.Tensor): Scaled tensor to revert
            
        Returns:
            torch.Tensor: Tensor in original scale
            
        Raises:
            ValueError: If an unknown scaling method is specified
        """
        if self.method == 'standard':
            return data * self.std + self.mean
        elif self.method == 'minmax':
            return data * (self.max - self.min) + self.min
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
    def fit_transform(self, data:torch.Tensor):
        """
        Fit the scaler to data and then transform it in one step.
        
        Args:
            data (torch.Tensor): Input tensor to fit and transform
            
        Returns:
            torch.Tensor: Scaled tensor
        """
        self.fit(data)
        return self.transform(data)

def resize_images_in_folder(imgs_location:List[str], target_size:Tuple[int, int]):
    """
    Resize images in a folder to a target size.
    
    This function processes a list of image file paths and resizes each image
    to the specified target dimensions, saving the resized images back to their
    original locations.
    
    Args:
        imgs_location: List of image file paths
        target_size: Tuple of target width and height (width, height)
        
    Returns:
        None: The function modifies images in-place
    """
    
    total_images = len(imgs_location)

    for img_path in tqdm.tqdm(imgs_location, desc="Resizing images", total=total_images):
        try:
            img = Image.open(img_path)
            if img.size != target_size:
                img = img.resize(target_size)
                img.save(img_path)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue