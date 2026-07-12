# Project 1 Corrections

Use this alongside `260326 - project1_neural_networks.docx.pdf`.

I am keeping this as an errata sheet rather than a complete `transformer.py`, because the point of Project 1 is for you to write and understand the internals yourself.

## Step 1 - Project Location

The PDF suggests `~/projects/transformer-from-scratch`. For TARQ, use this folder instead:

```bash
cd /Users/ompatnaik/Documents/Code/tarq/project1
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Why: keeps all TARQ work under one portfolio folder.

## Step 3 - Data Download

The PDF line break can corrupt the URL. Use this exact command:

```bash
curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

After download, check:

```bash
wc -c input.txt
```

You should see roughly 1.1 million bytes.

## Step 3 - File Encoding

Use an explicit encoding when reading the text:

```python
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
```

Why: avoids platform-dependent text decoding surprises.

## Step 3 - Device Choice

The PDF line is basically correct:

```python
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
```

For debugging, it is okay to temporarily force CPU:

```python
DEVICE = "cpu"
```

Why: MPS is faster on your Mac, but CPU errors are sometimes simpler while learning.

## Step 4 - Embedding Test Device Mismatch

The PDF test creates `emb` on CPU, but `get_batch()` returns `x` on `DEVICE`. On your Mac, `DEVICE` is likely `mps`, so the original test fails.

Replace:

```python
emb = Embeddings(VOCAB, EMBED_DIM)
x, _ = get_batch("train")
out = emb(x)
```

with:

```python
emb = Embeddings(VOCAB, EMBED_DIM).to(DEVICE)
x, _ = get_batch("train")
out = emb(x)
print(out.shape, out.device)
```

Expected:

```text
torch.Size([32, 128, 128]) mps:0
```

or CPU if you forced `DEVICE = "cpu"`.

## Step 5 - Positional Encoding Robustness

The current hyperparameter `EMBED_DIM = 128` is even, so the PDF code works. Add this guard anyway:

```python
assert embed_dim % 2 == 0, "sinusoidal positional encoding expects an even embed_dim"
```

Put it at the start of `PositionalEncoding.__init__`.

Why: if you later try an odd embedding size, the sine/cosine assignment can break.

## Step 5 - Positional Encoding Device

The PDF correctly uses:

```python
self.register_buffer("pe", pe.unsqueeze(0))
```

Do not replace this with `self.pe = ...`.

Why: registered buffers move with `model.to(DEVICE)`. Plain tensor attributes do not, which would create another CPU/MPS mismatch later.

## Step 6 - Self-Attention Mask

The PDF creates the causal mask inside `forward()` with:

```python
mask = torch.tril(torch.ones(T, T, device=x.device))
```

This is correct and device-safe. It is slightly inefficient but fine for learning.

Important shape check:

```python
sa = SelfAttention(EMBED_DIM, EMBED_DIM // NUM_HEADS).to(DEVICE)
dummy = torch.randn(BATCH_SIZE, CONTEXT_LEN, EMBED_DIM, device=DEVICE)
print(sa(dummy).shape)
```

Expected:

```text
torch.Size([32, 128, 32])
```

## Step 7 - Multi-Head Attention Shape Check

After `MultiHeadAttention`, test:

```python
mha = MultiHeadAttention(EMBED_DIM, NUM_HEADS).to(DEVICE)
dummy = torch.randn(BATCH_SIZE, CONTEXT_LEN, EMBED_DIM, device=DEVICE)
print(mha(dummy).shape)
```

Expected:

```text
torch.Size([32, 128, 128])
```

Why: each head returns 32 features, four heads concatenate back to 128.

## Step 10 - Cross-Entropy Flattening

The PDF uses `.view(...)`:

```python
loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
```

Use `.reshape(...)` instead:

```python
loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
```

Why: `.view()` only works on contiguous tensors. `.reshape()` is safer and teaches the same idea: flatten batch and time into one prediction axis.

## Step 10 - Generation Mode

The PDF's `generate()` method calls `self.eval()` at the start and `self.train()` at the end. That is okay if you only generate during/after training.

A slightly safer pattern is to remember the previous mode:

```python
was_training = self.training
self.eval()
# generation loop
if was_training:
    self.train()
```

Why: if you ever call `generate()` while the model was already in eval mode, it should stay in eval mode afterward.

## Step 11 - Estimate Loss Tensor Formatting

The PDF stores losses in CPU tensors:

```python
losses = torch.zeros(50)
```

That works because it stores `loss.item()`. To make the returned dictionary cleaner, use Python floats:

```python
out[split] = losses.mean().item()
```

Then update the training loop:

```python
ppl = torch.exp(torch.tensor(losses["val"])).item()
train_losses.append(losses["train"])
val_losses.append(losses["val"])
```

and format with:

```python
f"train={losses['train']:.4f} "
f"val={losses['val']:.4f} "
```

Why: avoids mixing scalar tensors and Python floats in logs/lists.

## Step 11 - First Training Run

Before a full 5000-step run, temporarily reduce:

```python
TRAIN_STEPS = 20
EVAL_INTERVAL = 5
```

Run:

```bash
python transformer.py
```

Only after the script completes, restore:

```python
TRAIN_STEPS = 5000
EVAL_INTERVAL = 500
```

Why: this catches shape/device bugs quickly before you spend time training.

## Step 13 - Plotting on macOS

Saving the plot is enough:

```python
plt.savefig("training_curves.png", dpi=160)
```

If you add `plt.show()`, the script may pause waiting for a window depending on the environment.

## Final Sanity Checklist

Before the real run:

- `input.txt` exists in this folder.
- `DEVICE` prints `mps` or `cpu`.
- Step 4 embedding test prints `(32, 128, 128)` on the same device as `x`.
- Self-attention test prints `(32, 128, 32)`.
- Multi-head attention test prints `(32, 128, 128)`.
- The full model returns logits shaped `(32, 128, VOCAB)` and a non-None scalar loss.
- A 20-step smoke test runs before the 5000-step training run.

When something breaks, identify the failing boundary first: data batch, embedding, positional encoding, attention, block, full model, or training loop.
