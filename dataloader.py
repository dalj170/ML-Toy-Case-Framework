import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    """ A simple Dataset class to handle the generation and use of f(x) = 1/2x**2 data"""
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

