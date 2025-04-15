import torch


class Generator:
    def __init__():
        pass

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
