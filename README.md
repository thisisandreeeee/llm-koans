# LLM Koans

A hands-on set of practical Python koans for building intuition about how LLMs work *and* how they get served in the real world.

The early koans still teach the core mechanics behind modern LLMs:

- **Query** = what this token is looking for.
- **Key** = how another token advertises that it is relevant.
- **Value** = the information retrieved if that token is attended to.
- **Dot product** = an alignment/similarity score between a query and a key.
- **Softmax** = converts raw scores into positive weights that sum to 1.
- **Context vector** = a weighted blend of values; the token after gathering context.
- **Encoder/decoder blocks** = the classic Transformer building blocks.
- **Training update** = the step that changes parameters, not the temporary Q/K/V activations.

The repo now broadens from model internals into production-shaped LLM work:

- serving generation behind a small FastAPI app
- validating context-window and output-token budgets
- routing requests across healthy backends
- micro-batching under token limits
- deciding which failures are safe to retry

The attention exercises are inspired by the step-by-step structure in Sebastian Raschka's article, "Understanding and Coding the Self-Attention Mechanism of Large Language Models From Scratch":

https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html

## Repository layout

```text
llm-koans/
├── src/llm_koans/                 # You edit the focused koan modules here
│   ├── shapes.py                  # Koan 01
│   ├── attention.py               # Koans 02-04
│   ├── masks.py                   # Koan 05
│   ├── blocks.py                  # Koan 06
│   ├── training.py                # Koan 07
│   ├── deployment.py              # Koan 08
│   └── koans.py                   # Stable public API used by tests
├── tests/                         # Tests verify each koan
├── tools/check.py                  # Convenience test runner
├── README.md
├── KOANS.md                       # Learning path and hints
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
cd attention-transformer-koans  # repository name on GitHub for now
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e .
```

## How to work through the koans

Run all tests:

```bash
pytest
```

At first, many tests will fail because the focused modules in `src/llm_koans/` contain `TODO` placeholders.

Work through the tests in order:

```bash
pytest tests/test_01_shapes_and_projections.py -q
pytest tests/test_02_attention_scores.py -q
pytest tests/test_03_self_attention.py -q
pytest tests/test_04_multihead_attention.py -q
pytest tests/test_05_masks_and_decoder_attention.py -q
pytest tests/test_06_encoder_decoder_blocks.py -q
pytest tests/test_07_training_updates.py -q
pytest tests/test_08_llm_deployment.py -q
```

Or use the helper:

```bash
python tools/check.py
python tools/check.py 03
python tools/check.py 08
```

## Suggested learning loop

1. Open the failing test.
2. Read the test name and comments.
3. Implement only the function needed for that test in the matching focused module.
4. Run the test again.
5. Move to the next test file.

This is intentionally not a polished library. It is a learning repo. The tests are the teacher.

## Shape convention used here

Most tensor functions use the practical PyTorch-friendly convention:

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

This matches the practical mental model: keep tokens as rows, then use `X @ W.T` for projection-matrix examples.
