import torch
import torch.nn as nn

import time


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, in_dim, out_dim, context_size, dropout, qkv_bias=False):
        super().__init__()

        self.head_num = head_num
        self.head_dim = out_dim // head_num
        self.out_dim = out_dim
        self.context_size = context_size
        self.w_q = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.w_k = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.w_v = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        # context_size == token_num, because we need to calculate context_vec for all tokens,
        # the result size is (token_num * token_num)
        self.register_buffer('mask', torch.triu(
            torch.ones(context_size, context_size), diagonal=1))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, inputs):
        b, token_num, token_vec_size = inputs.shape

        queries = self.w_q(inputs)
        keys = self.w_k(inputs)
        values = self.w_v(inputs)

        # Note:
        # out_dim = token_vec_size
        # out_dim = self.head_num * self.head_dim
        # modify dimension for multi-head matrix multiplication
        queries = queries.view(b, token_num, self.head_num, self.head_dim)
        keys = keys.view(b, token_num, self.head_num, self.head_dim)
        values = values.view(b, token_num, self.head_num, self.head_dim)

        # transpose (b,token_num,head_num, head_dim)
        # to : (b,head_num, token_num, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (b,head_num, token_num, head_dim) @ (b,head_num, head_dim, token_num)
        att_scores = queries @ keys.transpose(2, 3)

        # core concept of causal: remove the scores of afterwards words to force model to only attention to previous words
        # it replace next word scores with -inf to nuterilize the effect of next words scores in softmax function
        # Note: e^-inf == 0
        mask_bool = self.mask.bool()[: token_num, :token_num]
        att_scores.masked_fill_(mask_bool, -torch.inf)

        # scale down attention scores to stabilize the learning process by avoiding the large  value of dot product.
        # By dividing the attention scores by the square root of the key vector dimension, \
        # we strike a balance between capturing meaningful relationships and preventing numerical instability,\
        # allowing the model to effectively attend to the relevant information in the input sequence.
        attention_weights = torch.softmax(
            att_scores / keys.shape[2] ** 0.5, dim=-1)

        # apply dropout
        attention_weights = self.dropout(attention_weights)

        # transpose to (b, token_num, head_num, head_dim)
        context_vec = (attention_weights @  values).transpose(1, 2)
        # combining heads by concatinating the head outputs -> (b, token_num, out_dim)
        context_vec = context_vec.contiguous().view(b, token_num, self.out_dim)

        # adding optional linear projection
        context_vec = self.out_proj(context_vec)

        return context_vec


if __name__ == "__main__":
    inputs = torch.rand((2048, 1024))
    batch = torch.stack([inputs, inputs, inputs, inputs, inputs, inputs])
    print(inputs.shape)
    print(batch.shape)
    b, token_num, token_vec_size = batch.shape
    d_out = 16
    head_num = 8
    multi_causal_att = MultiHeadAttention(
        head_num, token_vec_size, d_out, token_num, 0.0, False)
    context_vec = multi_causal_att(batch)
    print(context_vec.shape)
