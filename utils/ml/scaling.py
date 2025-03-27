import torch

class CustomScaler:
    def __init__(self, method: str = 'standard'):
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    def fit(self, data:torch.Tensor):
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
        if self.method == 'standard':
            return (data - self.mean) / (self.std + 1e-8)
        elif self.method == 'minmax':
            return (data - self.min) / (self.max - self.min + 1e-8)
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
    def inverse_transform(self, data:torch.Tensor):
        if self.method == 'standard':
            return data * self.std + self.mean
        elif self.method == 'minmax':
            return data * (self.max - self.min) + self.min
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
    def fit_transform(self, data:torch.Tensor):
        self.fit(data)
        return self.transform(data)