import torch 
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


"""
Tokenize 

Purpose
"""
def tokenize(path): 
    with open(path, 'r', encoding='utf-8') as f: 
        text = f.read()
        chars = sorted(list(set(text)))
        stoi = {ch:i for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        data = torch.tensor(encode(text))
        n = int(0.9*len(data))
        train_data, val_data = data[:n], data[n:]

        itos = {i:ch for i, ch in enumerate(chars)}
        decode = lambda l: [itos[i] for i in l]

        return train_data, val_data, chars, decode

class BigramLanguageModel(nn.Module): 
    def __init__(self, vocab_size): 
        super().__init__() 
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None): 
        # idx and targets are both (B, T) tensor of ints
        logits = self.token_embedding_table(idx) # (B,T,C), batch (4) Time (block_size)  C (vocab size)

        if targets is None: 
            loss = 0
        else: 
            # cross entropy expects (B, C, T), channel as dim=1
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens): 
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens): 
            # get preds
            logits, _ = self(idx)
            # focus on last time step 
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probas 
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution 
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to next running seq
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx

    
if __name__ == "__main__": 
    train_data, val_data, vocab, decode = tokenize('input.txt')
    
    block_size = 8 
    print(train_data[:block_size+1])

    # exploring the time dimension (varying context length) so model can work for any context length up to block size limit
    x = train_data[:block_size]
    y = train_data[1:block_size+1]
    for t in range(block_size): 
        context = x[:t+1]
        target = y[t]
        print(f'when input is {context} the target: {target}')

    batch_size = 4 # how many independent sequences to process in parallel? 
    block_size = 8 # what is the maximum context length for prediction?

    def get_batch(split="train"): 
        # generate small batch of data input x and target y
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size, )) # randomly sample from valid indices
        x = torch.stack([data[i:i+block_size] for i in ix]) # retrieve valid character blocks, training will account for diff context lengths
        y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # offset by 1 since those are the targets for each context length up to block_size
        return x, y
    

    xb, yb = get_batch('train')

    vocab_size = len(vocab)
    m = BigramLanguageModel(vocab_size)
    
    # training script
    batch_size = 32
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for steps in range(100): 
        # sample a batch 
        xb, yb = get_batch('train')

        # eval loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generation
    print(''.join(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long),max_new_tokens=100)[0].tolist())))

    

    
