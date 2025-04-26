
import os
import sys
import torch
import tiktoken
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(parent_dir)


class Generator:
    def __init__(self, model=None, tokenizer=None, config=None):
        self.model = model
        self.tokenizer = tokenizer

        if config is None:
            self.config = self.get_default_config()
        else:
            self.config = config

    def simple_generate(self, model, idx, context_size, max_new_token):
        for _ in range(max_new_token):
            with torch.no_grad():
                current_idx = idx[:, -context_size:]
                logits = model(current_idx)
                logits = logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_idx = torch.argmax(probs, dim=-1, keepdim=True)
                idx = torch.cat((idx, next_idx), dim=1)

        return idx

    def text_to_tokenIds(self, text, tokenizer: tiktoken.core.Encoding = None):
        if tokenizer is None:
            tokenizer = self.tokenizer

        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensot = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensot

    def tokenIds_to_text(self, tokenIds, tokenizer: tiktoken.core.Encoding = None):
        if tokenizer is None:
            tokenizer = self.tokenizer
        flat = tokenIds.squeeze(0)
        decoded_tokens = tokenizer.decode(flat.tolist())
        return decoded_tokens

    def generate(self, prompt):

        token_ids = self.text_to_tokenIds(prompt)

        generated_token_ids = self.simple_generate(model=self.model,
                                                   context_size=self.config['context_length'],
                                                   idx=token_ids,
                                                   max_new_token=self.config['max_new_token'])

        text = self.tokenIds_to_text(
            generated_token_ids)

        return text

    def get_default_config():
        return {
            "vocab_size": 50257,
            "context_length": 256,
            "emb_dim": 768,
            "n_heads": 12,
            "n_layers": 12,
            "drop_rate": 0.1,
            "qkv_bias": False,
            "max_new_token": 512
        }

    def get_default_tokenizer():
        return tiktoken.get_encoding("gpt2")


if __name__ == "__main__":
    from architectures.gpt.gpt import GPTModel
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "max_new_token": 10
    }
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    generator = Generator(model=model,
                          tokenizer=tokenizer,
                          config=GPT_CONFIG_124M)

    start_context = "Hello LLMs' world"

    text = generator.generate(prompt=start_context)
    print(text)
