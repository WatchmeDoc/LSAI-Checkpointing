import torch.nn.functional as F

def attention(q, k, v, is_causal):
    output = F.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal
    )
    return output