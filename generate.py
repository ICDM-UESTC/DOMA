import warnings
warnings.filterwarnings("ignore")
from transformers.utils import logging
logging.set_verbosity_error()  
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn.functional as F

def nll_loss(logits, oracle_tokens, gate=None, ignore_index=-100):
    B, L, V = logits.shape
    if oracle_tokens.dim() == 1:
        oracle_tokens = oracle_tokens.unsqueeze(0)  
    if gate is not None:
        if gate.dim() == 1:
            gate = gate.unsqueeze(0).expand(B, -1)  #
        oracle_tokens = oracle_tokens.clone()
        oracle_tokens[gate == 0] = ignore_index
    logits_flat = logits.view(B * L, V)
    tokens_flat = oracle_tokens.view(B * L)
    loss = F.cross_entropy(logits_flat, tokens_flat, ignore_index=ignore_index)
    return loss


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=1.0,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks 
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) 
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) 
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    return x


@torch.no_grad() # comment out if training
def generate_ap(model, tokenizer, prompt, pred_asr, oracle_asr, ap,
                steps=32, gen_length=64, block_length=64,
                temperature=1.0, cfg_scale=0., remasking='low_confidence', mask_id=126336,
                mode="train"):
    device = next(model.parameters()).device
    input_ids = prompt.clone() if isinstance(prompt, torch.Tensor) else torch.tensor(tokenizer(prompt)['input_ids']).unsqueeze(0).to(device)
    prompt_len = input_ids.shape[1]
    asr_ids = pred_asr.clone() if isinstance(pred_asr, torch.Tensor) else torch.tensor(tokenizer(pred_asr)['input_ids']).unsqueeze(0).to(device)
    asr_len = asr_ids.shape[1]
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    x_=x.clone()
    x[:, prompt_len:prompt_len + asr_len] = asr_ids
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    embedding_layer = model.get_input_embeddings()
    x_embed = embedding_layer(x).float().detach()
    response_embed = x_embed[:, prompt_len:]  
    asr_binary_gate = ap(response_embed) 
    asr_binary_gate[asr_len:] = 1
    num_protect = (asr_binary_gate==0).sum().item()
    gate_for_asr = asr_binary_gate.bool().unsqueeze(0)  
    x[:, :prompt_len + gen_length][:, prompt_len:prompt_len + gen_length] = torch.where(
        gate_for_asr,  
        mask_id,       
        x[:, :prompt_len + gen_length][:, prompt_len:prompt_len + gen_length] 
    )
    asr_binary_gate = torch.cat([torch.ones(prompt_len, device=asr_binary_gate.device), asr_binary_gate], dim=0)
    final_logits = torch.zeros(1, prompt_len + gen_length, model.config.vocab_size, device=device)

    for num_block in range(num_blocks):
        start = prompt_len + num_block * block_length
        end = start + block_length
        block_mask_index = (x_[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            if i < num_protect:
                continue
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[:, :prompt_len] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            logits = logits * asr_binary_gate.unsqueeze(-1)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            logits_with_noise = logits_with_noise * asr_binary_gate.unsqueeze(-1)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                x0_p = x0_p * asr_binary_gate.unsqueeze(0)
                x0_p = torch.where(x0_p==0, torch.tensor(-np.inf, device=x0_p.device), x0_p)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            x0_p[:, prompt_len + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):  
                _, select_index = torch.topk(confidence[j, start:end], k=num_transfer_tokens[j, i])
                transfer_index[j, start:end] = False 
                transfer_index[j, start:end][select_index] = True
            x[transfer_index] = x0[transfer_index]
            final_logits[:, start:end, :] = logits[:, start:end, :]
    if mode == "train":
        loss_asr = nll_loss(final_logits[:, prompt_len:, :], oracle_asr, gate=asr_binary_gate[prompt_len:]) if oracle_asr is not None else None
        return x, loss_asr
    else:
        return x