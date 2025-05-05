import numpy as np
import torch


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                         "Right: {right.shape}"
                         )
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.positional_embedding.weight = assign(
        gpt.positional_embedding.weight, params['wpe'])
    gpt.token_embedding.weight = assign(
        gpt.token_embedding.weight, params['wte'])

    for b in range(len(params["blocks"])):
        # load weights
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)

        gpt.transformer_blocks[b].multi_attention.w_q.weight = assign(
            gpt.transformer_blocks[b].multi_attention.w_q.weight, q_w.T)
        gpt.transformer_blocks[b].multi_attention.w_k.weight = assign(
            gpt.transformer_blocks[b].multi_attention.w_k.weight, k_w.T)
        gpt.transformer_blocks[b].multi_attention.w_v.weight = assign(
            gpt.transformer_blocks[b].multi_attention.w_v.weight, v_w.T)

        # load bias
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)

        gpt.transformer_blocks[b].multi_attention.w_q.bias = assign(
            gpt.transformer_blocks[b].multi_attention.w_q.bias, q_b)
        gpt.transformer_blocks[b].multi_attention.w_k.bias = assign(
            gpt.transformer_blocks[b].multi_attention.w_k.bias, k_b)
        gpt.transformer_blocks[b].multi_attention.w_v.bias = assign(
            gpt.transformer_blocks[b].multi_attention.w_v.bias, v_b)

        gpt.transformer_blocks[b].multi_attention.out_proj.weight = assign(
            gpt.transformer_blocks[b].multi_attention.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].multi_attention.out_proj.bias = assign(
            gpt.transformer_blocks[b].multi_attention.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.transformer_blocks[b].feed_forward.layers[0].weight = assign(
            gpt.transformer_blocks[b].feed_forward.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_blocks[b].feed_forward.layers[0].bias = assign(
            gpt.transformer_blocks[b].feed_forward.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformer_blocks[b].feed_forward.layers[2].weight = assign(
            gpt.transformer_blocks[b].feed_forward.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].feed_forward.layers[2].bias = assign(
            gpt.transformer_blocks[b].feed_forward.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # load layer normalization weights
        gpt.transformer_blocks[b].norm1.scale = assign(
            gpt.transformer_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformer_blocks[b].norm1.shift = assign(
            gpt.transformer_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformer_blocks[b].norm2.scale = assign(
            gpt.transformer_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformer_blocks[b].norm2.shift = assign(
            gpt.transformer_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    # load final normalization layer weights
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

    return gpt
