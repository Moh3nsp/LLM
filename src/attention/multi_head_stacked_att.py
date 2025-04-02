import time
import torch
import torch.nn as nn
from causal_attention import CausalAttention


class MultiHeadStacked(nn.Module):
    def __init__(self, head_num, in_dim, out_dim, context_vec_size, dropout, bias):
        super().__init__()
        self.head_num = head_num
        self.heads = nn.ModuleList([CausalAttention(input_dim=in_dim,
                                                    output_dim=out_dim,
                                                    context_length=context_vec_size,
                                                    dropout=dropout,
                                                    qkv_bias=bias) for i in range(head_num)])

    def forward(self, inputs):
        return torch.concat([head(inputs) for head in self.heads], dim=-1)


if __name__ == "__main__":
    inputs = torch.rand((2048, 1024))
    batch = torch.stack([inputs, inputs, inputs, inputs, inputs, inputs])
    print(inputs.shape)
    print(batch.shape)
    b, token_num, token_vec_size = batch.shape
    d_out = 16
    head_num = 8
    multi_causal_att = MultiHeadStacked(
        head_num, token_vec_size, d_out, token_num, 0.0, False)
    context_vec = multi_causal_att(batch)
    print(context_vec.shape)
