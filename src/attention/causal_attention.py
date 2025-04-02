import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, input_dim, output_dim, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.output_dim = output_dim
        self.w_query = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.w_key = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.w_value = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(
            torch.ones(context_length, context_length), diagonal=1))

    def forward(self, inputs):
        token_num = inputs.shape[1]
        quesries = self.w_query(inputs)
        keys = self.w_key(inputs)
        values = self.w_value(inputs)

        attention_scores = quesries @ keys.transpose(1, 2)

        # Masked_attention
        # core concept of causal: remove the scores of afterwards words to force model to only attention to previous words
        # it replace next word scores with -inf to nuterilize the effect of next words scores in softmax function
        # Note: e^-inf == 0
        attention_scores.masked_fill_(
            self.mask.bool()[:token_num, :token_num], -torch.inf)

        #! Scale down
        # scale down attention scores by keys' dimension square root (sqrt or **0.5)
        attention_weights = torch.softmax(
            attention_scores / torch.math.sqrt(keys.shape[-1]), dim=-1)

        # Dropout
        # apply dropout to better generalization
        attention_weights = self.dropout(attention_weights)

        # calculate final context_vector for all tokens
        context_vec = attention_weights @ values
        return context_vec


if __name__ == '__main__':
    inputs = torch.rand((10, 6))
    batch = torch.stack([inputs, inputs, inputs, inputs, inputs, inputs])
    print(inputs.shape)
    print(batch.shape)
    b, token_num, token_vec_size = batch.shape
    d_out = 4
    causal_att = CausalAttention(token_vec_size, d_out, token_num, 0.0)
    context_vec = causal_att(batch)
    print(context_vec.shape)
