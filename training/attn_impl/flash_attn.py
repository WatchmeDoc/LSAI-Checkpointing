from flash_attn_interface import flash_attn_func

def attention(q, k, v, is_causal):
    output, _ = flash_attn_func(q, k, v, causal=is_causal)
    return output