# Character-Level Transformer From Scratch

This project implements a small GPT-style character language model in PyTorch. The goal is not to build a production language model, but to understand the core machinery behind transformers: embeddings, positional encoding, causal self-attention, multi-head attention, residual connections, LayerNorm, training loops, validation loss, perplexity, and autoregressive generation.

It is part of the TARQ learning sequence.

## What It Builds

- Character-level tokenizer over Tiny Shakespeare
- Train/validation batching for next-character prediction
- Token embeddings
- Fixed sinusoidal positional encoding
- Causal self-attention with a lower-triangular mask
- Multi-head attention
- Transformer blocks with pre-LayerNorm and residual connections
- Feed-forward network with GELU
- Weight-tied language-model head
- AdamW training loop with gradient clipping
- Validation loss, perplexity, and training curve plot
- Text generation from the trained model

## Results

The model trains from an initial loss of about `log(vocab_size)` and reaches validation loss around the expected range for a small character-level model.

![Training curve](training_curves.png)

## Setup

```bash
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Run

```bash
python transformer.py
```

The script trains the model, prints train/validation loss and perplexity, generates a text sample, and saves `training_curves.png`.

## Files

- `transformer.py` - main implementation
- `WRITEUP.md` - explanation connecting this model to mBERT/MuRIL and transformer concepts
- `CORRECTIONS.md` - learning notes and fixes used while working through the curriculum PDF
- `requirements.txt` - Python dependencies

## Learning Notes

This is intentionally small. It is a character-level decoder-only transformer trained on a tiny corpus, so generated text will be rough. The value of the project is being able to explain every tensor shape and every training step.

The key mental model:

```text
text -> character IDs -> batches -> embeddings -> positional encoding
     -> transformer blocks -> logits -> cross-entropy loss -> gradients
```
