
import torch.nn.functional as F
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

    def generateV2(self, model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

        # For-loop is the same as before: Get logits, and only focus on last time step
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]

            # New: Filter logits with top_k sampling
            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(
                    float("-inf")).to(logits.device), logits)

            # New: Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Apply softmax to get probabilities
                # (batch_size, context_len)
                probs = torch.softmax(logits, dim=-1)

                # Sample from the distribution
                idx_next = torch.multinomial(
                    probs, num_samples=1)  # (batch_size, 1)

            # Otherwise same as before: get idx of the vocab entry with the highest logits value
            else:
                idx_next = torch.argmax(
                    logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break

            # Same as before: append sampled index to the running sequence
            # (batch_size, num_tokens+1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

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
