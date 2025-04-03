
import torch
import torch.nn as nn

from transformer.normalization import LayerNormalization
from transformer.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config['vocab_size'], config['emb_dim'])

        self.positional_embedding = nn.Embedding(
            config['context_length'], config['emb_dim'])

        self.dropout = nn.Dropout(config['drop_rate'])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['n_layers'])])

        self.final_norm = LayerNormalization(config['emb_dim'])

        self.out_head = nn.Linear(
            config['emb_dim'], config['vocab_size'], bias=False)

    def forward(self, in_idx):
        batch, seq_len = in_idx.shape
        toks_emb = self.token_embedding(in_idx)
        pos_emb = self.positional_embedding(
            torch.arange(seq_len, device=in_idx.device))

        x = toks_emb+pos_emb
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
