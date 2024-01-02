import torch
import torch.nn as nn
import torch.nn.functional as F

block_size = 8
n_embd = 32

class Head(nn.Module):
    def __init__(self, head_size): 
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x): 
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute scaled attn scores
        wei = q * k.transpose(1, 2) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T], float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # perform weighted aggregation
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, C) = (B, T, C)

        return out
