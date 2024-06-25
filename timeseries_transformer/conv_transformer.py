import math, torch

import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
num_heads = 6
n_layers = 6
dropout = 0.2

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
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # not a parameter of the model, so create it with register_buffer.
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        # attention affinities
        weights = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # This is the form of guaranteeing future time tokens do not communicate with former tokens.
        weights = F.softmax(weights, dim=-1)
        
        weights = self.dropout(weights)

        v = self.value(x)
        out = weights @ v # (B,T,T) @ (B,T,head_size) = (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication and computation (MultiHead Attention + Feed Forward)"""

    def __init__(self, num_heads, n_embd) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.sa = MultiHeadAttention(num_heads, n_embd // num_heads)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # (C, N)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd) # (T, N)
        # self.sa_head = Head(n_embd)
        # self.sa_heads = MultiHeadAttention(4, n_embd//4)
        # self.ffwd = FeedForward(n_embd)
        
        self.blocks = nn.Sequential(*[Block(num_heads, n_embd) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size) # (N, C)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # (B,T,N) stands for batch x time x vocab_size
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device)) # (T, N)
        x = tok_emb + pos_emb # (B, T, N) + (T, N) broadcasting
        # x = self.sa_head(x)
        # x = self.sa_heads(x)
        # x = self.ffwd(x)

        x = self.blocks(x)
        x = self.layer_norm(x)        
        logits = self.lm_head(x) # (B, T, C)

        if targets is None:
            loss = None
        else:
            # stretching logits and targets tensors
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T) # equivalent to targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        self.eval()

        for i in range(max_new_tokens):
            idx_context = idx[:, -block_size:].reshape(1, -1) # Feed at most context window (length = block_size) to the model

            logits, _ = self(idx_context) # B,T,C
            logits = logits[:, -1, :] # B,C
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            idx = torch.cat((idx, idx_next), dim=-1)
        
        return idx


model = BigramLanguageModel_v2()
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

# Saving model and scripted model
torch.save(model, './nano-gpt.pt')

# model_scripted = torch.jit.script(model)
# model_scripted.save('./nano-gpt-scripted.pt') --> Variáveis globais como device devem ser registradas dentro do modelo.

# initial index
context = torch.zeros((1,1), dtype=torch.long).to(device) # B,T
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))