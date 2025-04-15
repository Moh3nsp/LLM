import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        """
        Note: 
        var = sqrt( ( sum( mean(X)-x_i)^2) forx_i in X ) / len(X) ) )
        with Bessel's correction -> denominator = len(x) -1, otherwise denominator = len(X)
        unbiased=False == doesn't apply Bassel's correction
        """
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
