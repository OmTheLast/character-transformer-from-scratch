# Project 1 Writeup

## Overview

This project builds a small GPT-style transformer from scratch in PyTorch. It uses character-level language modelling on Tiny Shakespeare: given a sequence of characters, the model predicts the next character at every position.

The purpose was not to produce fluent text. The purpose was to open the transformer black box and understand the machinery behind models such as GPT, mBERT, and MuRIL.

## Tokenization And Vocabulary

This model uses character-level tokenization. The vocabulary is only about 65 characters, so each input token is a single character ID.

mBERT and MuRIL use subword tokenization instead. mBERT uses WordPiece with a much larger multilingual vocabulary because it supports many languages at once. MuRIL is architecturally similar to BERT, but its pretraining data is better matched to Indian languages and transliterated text, which makes it stronger for Hinglish and related tasks.

The important idea is that tokenization decides what the model sees as a unit. This project uses characters because they are simple and make the training loop easy to inspect.

## Positional Encoding

Self-attention has no built-in sense of order. Without position information, the model would know which characters are present, but not where they occur in the sequence.

This project uses fixed sinusoidal positional encoding from the original Transformer paper:

```text
PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
```

Each position receives a deterministic vector made from sine and cosine waves at different frequencies. The token embedding and position vector are added together before entering the transformer blocks.

In neural-network terms, sinusoidal encoding is a fixed feature map from position index to vector. Learned positional embeddings are a trainable lookup table. Learned embeddings can work very well, but they add parameters and are limited to the positions included in the table. Sinusoidal encodings have zero trainable parameters and can be computed for positions beyond the original context length, although the model may not perform well beyond lengths it was trained on.

mBERT and MuRIL use learned absolute positional embeddings with a maximum context length, rather than fixed sinusoidal encoding.

## Attention

Each attention head creates three learned projections from the input:

- Query: what this token is looking for
- Key: what this token offers to be matched against
- Value: the information this token passes forward if attended to

Attention scores are computed with:

```text
QK^T / sqrt(head_dim)
```

The division by `sqrt(head_dim)` keeps the dot products from becoming too large, which prevents softmax from becoming overly sharp at initialization.

This model uses a causal mask, so each position can only attend to itself and earlier positions. That makes it a decoder-style model suitable for next-character generation.

## Transformer Block

Each block contains:

- LayerNorm
- Multi-head causal self-attention
- Residual connection
- LayerNorm
- Feed-forward network with GELU
- Residual connection

Residual connections help gradients flow through the network by giving them a direct identity path. LayerNorm stabilizes activations across the feature dimension.

## Training And Evaluation

The model is trained with cross-entropy loss. Logits shaped `(B, T, vocab_size)` are reshaped to `(B*T, vocab_size)` so every character position becomes one prediction example.

Validation loss and perplexity are measured during training. Perplexity is:

```text
perplexity = exp(loss)
```

It can be interpreted as the effective number of choices the model is uncertain between at each step.

The initial loss should be close to `log(vocab_size)`, because an untrained model is roughly guessing uniformly. In this project, small transformer-style initialization was necessary because the output head is weight-tied to the token embedding table. Without small initialization, logits became too large and the starting loss was incorrectly around 99.

## Connection To mBERT And MuRIL

This model is a causal decoder. It predicts the next token and uses a causal mask to prevent future-token leakage.

mBERT and MuRIL are encoder-style masked language models. They use bidirectional attention: each token can attend left and right. During pretraining, random tokens are masked and the model learns to reconstruct them from context.

The difference matches the task:

- GPT-style decoder: good for generation
- BERT-style encoder: good for understanding/classification

In the Hinglish hate-speech project, mBERT and MuRIL were not generating text. They were being fine-tuned for classification. A classification head maps the model's contextual representation to output labels. MuRIL performs better for Hinglish not because the architecture is radically different, but because its pretraining data better matches Indian-language and transliterated text patterns.

## Limitations

- Character-level modelling is simple but inefficient compared with subword tokenization.
- Tiny Shakespeare is small, so the model learns style patterns rather than broad language ability.
- The model is much smaller than production transformers.
- Generated text quality is not the main success metric; understanding the mechanism is.

