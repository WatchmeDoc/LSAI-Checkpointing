import os
import mmap
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from training.dataset import CollatorForCLM, ParquetDataset
from training.model import Transformer, TransformerModelArgs
from training.utils import build_lr_scheduler, clip_grad_norm_, get_args, get_num_params, get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype, set_seed, save_random_states, load_random_states, load_checkpoint
from copy import deepcopy
from checkpointing.pccheck.chk_monitor import Chk_monitor
from training.testing import compare_models_and_optimizers
from checkpointing.pccheck_utils import initialize, get_total_size, set_storage


device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")
train_ds = ParquetDataset("/capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet", tokenizer, 1024, 1*151)
train_collator = CollatorForCLM(1024, tokenizer.pad_token_id)
train_dl = DataLoader(train_ds, batch_size=1, collate_fn=train_collator)
train_dl_iterator = iter(train_dl)

state = load_random_states("./checkpointing/checkpoints")
train_step = state["step"]
model_config = TransformerModelArgs(
        dim=1024,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        vocab_size=tokenizer.vocab_size,
        seq_len=1024,
    )

with set_default_dtype(torch.float32):
    # CPU
    model = Transformer(model_config).to(device)
    model.train()


# Build Optimizers & LR Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=False)
lr_scheduler = build_lr_scheduler(optimizer, 10)
# for checkpoint
mp.set_start_method("spawn", force=True)
gpu_ar, model_size, optim_size = initialize(model, [optimizer])
total_size = model_size + optim_size
lr_scheduler.load_state_dict(state["lr_scheduler"])


max_async = 1
PAGE_SIZE = mmap.PAGESIZE
pr_data_offset = PAGE_SIZE * (max_async + 3) * 4
offset = pr_data_offset + PAGE_SIZE*4
with open("./checkpointing/checkpoints/checkpoint_pccheck.pt", "rb") as f:
        f.seek(offset, os.SEEK_SET)
        # convert numpy array into torch tensor
        payload = f.read()
        data = torch.frombuffer(payload, dtype=torch.float32)
data = torch.load("checkpointing/checkpoints/checkpoint_pccheck_ar.pt", map_location="cpu", weights_only=False)
start_idx = 0
for name, ref in model.named_parameters():
    end_idx = start_idx + ref.numel()
    my_ar = data[start_idx:end_idx].reshape(ref.shape)
    # print(f"Name: {name}, Shape: {ref.shape}, numel: {ref.numel()}, my_ar: {my_ar.shape}")
    prev_shape = ref.size()
    with torch.no_grad():
        ref.copy_(my_ar)
        # print(prev_shape, ref.shape, ref.data_ptr(), type(ref))
    start_idx += ref.numel()


# print(f"Optimizer learning rate: {optimizer.param_groups[0]['lr']}")
opt_state = optimizer.state_dict()
opt_state_copy = deepcopy(opt_state)
for name, _ in opt_state['state'].items():
    for k, ref in opt_state['state'][name].items():
        # print(k, ref.dtype)
        if (torch.is_tensor(ref)):
            end_idx = start_idx + ref.numel()
            t = data[start_idx:end_idx].reshape(ref.shape)
            opt_state_copy['state'][name][k].copy_(t)
            start_idx += ref.numel()
        elif (type(ref) == int or type(ref) == float):
            opt_state_copy['state'][name][k] = ref
            start_idx += 1

optimizer.load_state_dict(opt_state_copy)

ckpt = torch.load("./checkpointing/checkpoints/checkpoint_pccheck_test.pt", map_location="cpu", weights_only=False) # weights_only=False allows to load custom numpy pickles (UNSAFE)
model_b = deepcopy(model)
model.load_state_dict(ckpt["model"])

optim_b = deepcopy(optimizer)
optim_b.load_state_dict(ckpt["optimizer"])
print(f"Optimizer2 learning rate: {optimizer.param_groups[0]['lr']} vs {optim_b.param_groups[0]['lr']}")

compare_models_and_optimizers(model_a=model, optim_a=optimizer, model_b=model_b, optim_b=optim_b)

def train_loop(model, optimizer, train_step, train_dl_iterator):
    train_step += 1
    input_ids, labels = next(train_dl_iterator)
    num_items_in_batch = labels.ne(-100).sum()
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum")
    loss = loss / num_items_in_batch
    print(f"Loss: {loss.item()}")
    del logits
    loss.backward()
    # Clip gradients
    clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    lr_scheduler.step()

def initialize(model, optimizer_list, do_opt_step=True):
    if isinstance(model, dict):
        model_state = model
    else:
        model_state = model.state_dict()
    # initialize optimizer for realistic setups
    for optimizer in optimizer_list:
        opt_state = optimizer.state_dict()
        if len(opt_state['state']) == 0:
            for group in optimizer.param_groups:
                for p in group['params']:
                    p.grad = p.data.new(p.size())
        if do_opt_step:
            optimizer.step()
    model_size = 0
    for name, ref in model_state.items():
        if (torch.is_tensor(ref)):
            model_size += ref.numel()
        elif (type(ref) == int or type(ref) == float):
            model_size += 1
    opt_size = 0
    for optimizer in optimizer_list:
        opt_state = optimizer.state_dict()
        for name, _ in opt_state['state'].items():
            for k, ref in opt_state['state'][name].items():
                print(name, k, ref.dtype, ref.shape)
                if (torch.is_tensor(ref)):
                    opt_size += ref.numel()
                elif (type(ref) == int or type(ref) == float):
                    opt_size += 1
    total_size = model_size + opt_size
    gpu_ar = torch.zeros(total_size).cuda()
