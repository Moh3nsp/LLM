import torch
import torch.nn as nn
from .gelu_activation_function import GELU


class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            GELU(),
            nn.Linear(4*embedding_size, embedding_size)
        )

    def forward(self, x):
        return self.layers(x)
