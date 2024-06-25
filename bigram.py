import math, torch

import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

with open('input.txt','r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {c: i for i,c in enumerate(chars)}
itos = {i: c for i,c in enumerate(chars)}

encode = lambda s: [stoi[t] for t in s]
decode = lambda d: ''.join([itos[i] for i in d])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))

# Defined train/val datasets
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # define if computations are going to be processed in CPU or GPU
    return x, y


@torch.no_grad() # prevents pytorch from running backpropagation
def estimate_loss():
    out = {}
    model.train()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B,T,C) stands for batch x time x vocab_size
        B,T,C = logits.shape
        
        if targets is None:
            loss = None
        else:
            # stretching logits and targets tensors
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) # equivalent to targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            logits,_ = self(idx) # B,T,C
            logits = logits[:, -1, :] # B,C
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx


model = BigramLanguageModel(vocab_size)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for i in range(max_iters):

    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f'Step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

    x_batch, y_batch = get_batch('train')
    
    logits, loss = model(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    
    loss.backward()
    optimizer.step()

# initial index
context = torch.zeros((1,1), dtype=torch.long) # B,T
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))