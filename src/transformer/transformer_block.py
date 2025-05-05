import torch
import torch.nn as nn
from attention.multi_head import MultiHeadAttention
from transformer.feedforward import FeedForward
from transformer.normalization import LayerNormalization


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multi_attention = MultiHeadAttention(head_num=config['n_heads'],
                                                  in_dim=config['emb_dim'],
                                                  out_dim=config['emb_dim'],
                                                  context_size=config['context_length'],
                                                  dropout=config['drop_rate'],
                                                  qkv_bias=config['qkv_bias'])
        self.feed_forward = FeedForward(embedding_size=config['emb_dim'])

        self.norm1 = LayerNormalization(embedding_dim=config['emb_dim'])
        self.norm2 = LayerNormalization(embedding_dim=config['emb_dim'])
        self.droput = nn.Dropout(config['drop_rate'])

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.multi_attention(x)
        x = self.droput(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.droput(x)
        x = x + residual

        return x
