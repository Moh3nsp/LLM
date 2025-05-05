import tiktoken
import torch


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from architectures.gpt.gpt import GPTModel
    from text_generation.generate import Generator
    from gpt_download import download_and_load_gpt2
    from assign_weights import load_weights_into_gpt

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        'max_new_token': 100,
        "drop_rate": 0.1,
        "qkv_bias": True
    }

    model_name = "gpt2-medium (355M)"
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])

    settings, params = download_and_load_gpt2(
        model_size="355M", models_dir="gpt2-medium"
    )

    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()

    # params = torch.load('params.pth', weights_only=False, map_location=device)
    # settings = torch.load(
    #     'settings.pth', weights_only=False,  map_location=device)

    gpt = load_weights_into_gpt(gpt, params)
    gpt.to(device)

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    generator = Generator(model=gpt,
                          tokenizer=tokenizer,
                          config=GPT_CONFIG_124M)
    tokenIds = generator.text_to_tokenIds("Every effort moves you", tokenizer)
    # model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
    answer = generator.generateV2(model=gpt, idx=tokenIds, max_new_tokens=25, context_size=NEW_CONFIG["context_length"],
                                  top_k=50, temperature=1.0)

    answer = generator.tokenIds_to_text(answer, tokenizer)

    print("Output text:\n", answer)

   # for saving model
    torch.save(gpt, "gpt-medium.pth")
