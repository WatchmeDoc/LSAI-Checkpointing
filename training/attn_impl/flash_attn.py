from flash_attn_interface import flash_attn_func
import os
from training.utils import logger

if os.environ.get("VALIDATION_MODE") == "1":
    det_flash_attn = True
else:
    det_flash_attn = False

def attention(q, k, v, is_causal):
    output, _ = flash_attn_func(q, k, v, causal=is_causal, deterministic=det_flash_attn)
    return output