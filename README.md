# Attention & Transformer Koans

A hands-on set of PyTorch koans for building intuition about attention and Transformers.

The koans build on these ideas:

- **Query** = what this token is looking for.
- **Key** = how another token advertises that it is relevant.
- **Value** = the information retrieved if that token is attended to.
- **Dot product** = an alignment/similarity score between a query and a key.
- **Softmax** = converts raw scores into positive weights that sum to 1.
- **Context vector** = a weighted blend of values; the token after gathering context.
- **Encoder** = reads an input sequence and produces contextual token vectors.
- **Decoder** = generates output using masked self-attention and, in encoder-decoder models, cross-attention.

The exercises are inspired by the step-by-step structure in Sebastian Raschka's article, "Understanding and Coding the Self-Attention Mechanism of Large Language Models From Scratch":

https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html

## Repository layout

```text
attention-transformer-koans/
├── src/attention_koans/koans.py   # You edit this file
├── tests/                         # Tests verify each koan
├── tools/check.py                  # Convenience test runner
├── README.md
├── KOANS.md                       # Learning path and hints
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
cd attention-transformer-koans
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e .
```

## How to work through the koans

Run all tests:

```bash
pytest
```

At first, many tests will fail because `src/attention_koans/koans.py` contains `TODO` placeholders.

Work through the tests in order:

```bash
pytest tests/test_01_shapes_and_projections.py -q
pytest tests/test_02_attention_scores.py -q
pytest tests/test_03_self_attention.py -q
pytest tests/test_04_multihead_attention.py -q
pytest tests/test_05_masks_and_decoder_attention.py -q
pytest tests/test_06_encoder_decoder_blocks.py -q
pytest tests/test_07_training_updates.py -q
```

Or use the helper:

```bash
python tools/check.py
python tools/check.py 03
```

## Suggested learning loop

1. Open the failing test.
2. Read the test name and comments.
3. Implement only the function needed for that test.
4. Run the test again.
5. Move to the next test file.

This is intentionally not a polished library. It is a learning repo. The tests are the teacher.

## Shape convention used here

Most functions use the practical PyTorch-friendly convention:

```text
B = batch size
T = sequence length
D = d_model / embedding size
H = number of attention heads
Dh = per-head dimension

X:      (B, T, D)
Q/K/V:  (B, T, D) before splitting heads
heads:  (B, H, T, Dh)
scores: (B, H, T_query, T_key)
```

The basic single-sequence examples also use:

```text
X: (T, d)
W: (d_out, d_in)
X @ W.T -> (T, d_out)
```

This matches the mental model from our conversation: keep tokens as rows, then use `X @ W.T` for projections.
