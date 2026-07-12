import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import torch.optim as optim
import math

torch.manual_seed(42)

BATCH_SIZE    = 32 
CONTEXT_LEN   = 128 
EMBED_DIM     = 128 
NUM_HEADS     = 4 
NUM_LAYERS    = 3 
FF_DIM        = 512 
DROPOUT       = 0.1 
LEARNING_RATE = 3e-4 
TRAIN_STEPS   = 5000 
EVAL_INTERVAL = 500 
# Codex fix: evaluation frequency and evaluation sample count are separate.
# EVAL_INTERVAL controls how often we pause training; EVAL_ITERS controls
# how many random batches we average over when estimating train/val loss.
EVAL_ITERS    = 50
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Codex fix: explicit encoding keeps file reading stable across machines.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - CONTEXT_LEN, (BATCH_SIZE,))
    x = torch.stack([d[i:i+CONTEXT_LEN] for i in ix])
    y = torch.stack([d[i+1:i+1+CONTEXT_LEN] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ── EMBEDDINGS & POSITION MODULES ────────────────────────────────
class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        return self.lut(x)


#------Learned positional embeddings, as opposed to sinusoidal ones---------
#class PositionalEncoding(nn.Module):
#    def __init__(self, context_len, embed_dim):
#        super().__init__()
#        self.pe = nn.Embedding(context_len, embed_dim)
#    def forward(self, x):
#        positions = torch.arange(x.shape[1], device=DEVICE).unsqueeze(0)
#        return x + self.pe(positions)

#------Sinusoidal positional encoding, as per the original transformer paper---------
class PositionalEncoding(nn.Module):
    def __init__(self, context_len, embed_dim):
        super().__init__()
        #1 Create an empty tensor to hold the positional encodings
        pe = torch.zeros(context_len, embed_dim)

        #2 Generate positions column array vector: [context_len, 1]
        position = torch.arange(0, context_len, dtype=torch.float).unsqueeze(1)

        #3 Calculate the step scale terms for our denominators
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        #4 Fill even indices (0, 2, 4, ...) with sine values
        pe[:, 0::2] = torch.sin(position * div_term)

        #5 Fill odd indices (1, 3, 5, ...) with cosine values
        pe[:, 1::2] = torch.cos(position * div_term)

        #6 Save as buffer so PyTorch keeps it on the device withouth treating it as weights
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: [1, context_len, embed_dim]

    def forward(self, x):
        #Simply add the fixed coordinate wave values to our incoming token vectors
        #x shape: [B, T, C]
        return x + self.pe[:, :x.shape[1], :].to(x.device)  # Ensure the positional encoding is on the same device as x



# ── SELF-ATTENTION MODULES ───────────────────────────────────────
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(CONTEXT_LEN, CONTEXT_LEN)))
        self.dropout = nn.Dropout(DROPOUT) # Keeps the traits regularized

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei) # Drop attention paths randomly during training
        
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # Blend and drop out
        return out

# ── COMPUTATION MODULES ──────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, FF_DIM), # Using your FF_DIM = 512
            #nn.ReLU(),
            nn.GELU(), # Using GELU as per the original transformer paper
            nn.Linear(FF_DIM, n_embd),
            nn.Dropout(DROPOUT)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# ── THE COMPLETE CONSOLIDATED MODEL ──────────────────────────────
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = Embeddings(vocab_size, EMBED_DIM)
        self.position_embedding_table = PositionalEncoding(CONTEXT_LEN, EMBED_DIM)
        
        # Stacking the blocks sequentially using your NUM_LAYERS = 3
        self.blocks = nn.Sequential(*[Block(EMBED_DIM, n_head=NUM_HEADS) for _ in range(NUM_LAYERS)])
        
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, vocab_size, bias=False)

        # Codex fix: weight tying reuses the embedding table as the output
        # projection, so the embedding table must start with small values.
        # Without this init, logits start enormous and loss begins near 99.
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding_table.lut.weight

    # Codex fix: transformer-style small initialization keeps initial logits
    # close to zero, so random-model loss starts near log(vocab_size).
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Run through full processing pipe
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(tok_emb)
        x = self.blocks(pos_emb)
        x = self.ln_f(x)
        logits = self.lm_head(x) # Shape: [B, T, vocab_size]

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # Codex fix: reshape is safer than view if tensors are non-contiguous.
            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx shape is [B, T] representing the timeline array
        for _ in range(max_new_tokens):
            # Crop timeline so it doesn't overrun our CONTEXT_LEN positional map limits
            idx_cond = idx[:, -CONTEXT_LEN:]
            
            # Predict the logits using the forward pass
            logits, loss = self(idx_cond)
            
            # Crop to focus exclusively on the last chronological position
            logits = logits[:, -1, :] # Shapes down to [B, vocab_size]
            
            # Apply probability translation via Softmax
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the probability space
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Stitch the newly born character code onto our input string matrix
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx

#----Evaluation function for the model's performance on validation data
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        eval_losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            eval_losses[k] = loss.item()
        out[split] = eval_losses.mean()

    model.train()
    return out
            

# ── INITIALIZATION AND COMPILATION ──────────────────────────────
model = GPTLanguageModel().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE) # Using your LEARNING_RATE = 3e-4

total_params = sum(p.numel() for p in model.parameters())
print("\n==================================================")
print(f"Total Trainable Parameters: {total_params:,}")
print("==================================================\n")

print(f"Model initialized on device target: {DEVICE}")
print("--- Launching Text Generation Training Phase ---")

plot_steps = []
plot_train_loss = []
plot_val_loss = []

# Simple verification loop run
for step in range(TRAIN_STEPS):

    if step % EVAL_INTERVAL == 0 or step == TRAIN_STEPS - 1:
        losses = estimate_loss()

        #perplexity = e^loss
        train_perplexity = math.exp(losses['train'])
        val_perplexity = math.exp(losses['val'])

        print(f"Step {step:3d} | "
              f"Train Loss: {losses['train']:.4f} (Perp: {train_perplexity:.2f}) | "
              f"Val Loss: {losses['val']:.4f} (Perp: {val_perplexity:.2f})")

        plot_steps.append(step)
        plot_train_loss.append(losses['train'])
        plot_val_loss.append(losses['val'])
    xb, yb = get_batch('train')
    
    # Core execution pass
    logits, loss = model(xb, yb)
    
    # Calculus step tuning the weights
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping to avoid exploding gradients

    optimizer.step()
    
    if step % 100 == 0 or step == 499:
        print(f"Step {step:3d} | Matrix Training Loss: {loss.item():.4f}")

# ── RUN THE GENERATOR ───────────────────────────────────────────
print("\n--- Model Output Generation Validation ---")

# Seed the generator with index 0
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

# Call your new auto-regressive generation loop for 200 characters
generated_indices = model.generate(context, max_new_tokens=200)[0].tolist()

# Pass the output matrix through your existing decode lambda string generator
print("Generated Character Output Stream:")
print(decode(generated_indices))



print("\n---- Generating metric analysis plot ------")
plt.figure(figsize=(10, 6))
plt.plot(plot_steps, plot_train_loss, label ='Train Loss', linestyle='--')
plt.plot(plot_steps, plot_val_loss, label='Val Loss')
plt.xlabel("Training Optimization Steps")
plt.ylabel("Cross-Entropy Loss")
plt.title('Project 1 Evaluation Diagnostic Curve')
plt.legend()
plt.grid(True)
plt.savefig('training_curves.png')
print("Succesfully compiled and saved metrics data visualisation as 'training_curves.png' ")
