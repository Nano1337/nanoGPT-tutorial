import torch
import torch.nn as nn
import torch.nn.functional as F

vocab_size = 65
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

class LanguageModel(nn.Module): 
    def __init__(self): 
        super().__init__()  
        
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None): 
        B, T = idx.shape

        # idx and targets are both (B, T) dim
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.sa_head(x) # (B, T, C) 
        logits = self.lm_head(x) # (B, T, C)

        if targets is None: 
            loss = None 
        else: 
            logits = logits.view(B*T, -1)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens): 
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens): 
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] # limitation from pos emb size limit set initially
            # get preds
            logits, loss = self(idx_cond)
            # focus only on last time step
            logits = logits[:, -1, :] # (B, C)
            # apply softmax
            probs = F.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
