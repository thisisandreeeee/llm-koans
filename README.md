# LLM Koans

Learn how large language models work by completing small, focused PyTorch exercises.

A **koan** is a deliberately incomplete program that teaches one idea at a
time. Run its tests, study the failure, fill in the missing implementation,
and repeat until the tests pass. The short feedback loop turns concepts such
as attention, masking, and fine-tuning into code you can inspect and change.

This repository is a learning project, not a production LLM library. The tests
are the teacher, the source files are the workbook, and [`KOANS.md`](KOANS.md)
provides the lesson notes and hints.

## Learning path

The 12 koans build on one another:

| Stage                 | Koans | What you will learn                                                                                |
| --------------------- | ----- | -------------------------------------------------------------------------------------------------- |
| Tensor foundations    | 00–01 | Shapes, projections, dot products, scaled attention scores, and softmax                            |
| Transformer mechanics | 02–06 | Self-attention, multiple heads, causal masks, encoder/decoder blocks, and mixture of experts       |
| Models and training   | 07–08 | GPT-, BERT-, and T5-style models, next-token training, chat formatting, and supervised fine-tuning |
| Model adaptation      | 09–11 | LoRA adapters, knowledge distillation, and DPO preference tuning                                   |

See the complete sequence, learning goals, and function-level hints in [`KOANS.md`](KOANS.md).

## Getting started

You need Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/thisisandreeeee/llm-koans.git
cd llm-koans
git checkout main
uv sync
```

Start with the first koan:

```bash
uv run pytest tests/test_00_*.py
```

Koan 00 will fail because its functions contain `TODO` placeholders. That is your starting point.

## Working through the koans

For each koan:

1. Read its section in [`KOANS.md`](KOANS.md).
2. Open the matching test, such as [`tests/test_00_shapes_and_projections.py`](tests/test_00_shapes_and_projections.py).
3. Replace one `TODO` in the matching `src/llm_koans/koan_*.py` module.
4. Run that koan again and use the failure as your next clue.
5. Move to the next numbered koan when all of its tests pass.

Run one koan or the full suite with:

```bash
uv run pytest tests/test_03_*.py  # one koan
uv run pytest                     # every koan
```

Try to make the smallest change that satisfies the current test. If you get
stuck, revisit the test names, docstrings, and hints before looking at the answer.

## Reference solutions

Completed implementations live on the [`ref/model-answers`](https://github.com/thisisandreeeee/llm-koans/tree/ref/model-answers) branch.

You can inspect a solution without replacing your in-progress files:

```bash
git fetch origin
git show origin/ref/model-answers:src/llm_koans/koan_03_multihead_attention.py
```

Change `03_multihead_attention` to the koan you want to review. Avoid switching
your working tree to the solutions branch while solving the exercises, since
that branch replaces the `TODO`s with the completed code.

## Tensor shape convention

Most exercises use this PyTorch-friendly convention:

```text
B  = batch size
T  = sequence length
D  = model/embedding dimension
H  = number of attention heads
Dh = dimension per head

X:      (B, T, D)
Q/K/V:  (B, T, D) before splitting heads
heads:  (B, H, T, Dh)
scores: (B, H, T_query, T_key)
```

Remember that `torch.matmul` treats the final two dimensions as the matrix and
the earlier dimensions as batch dimensions:

```text
(..., rows, shared) @ (..., shared, cols) -> (..., rows, cols)
```

## Suggest a koan

Missing an LLM concept you would like to understand?

[Open an issue](https://github.com/thisisandreeeee/llm-koans/issues/new) and suggest the next koan.
Include the concept, what you hope to learn, and (if you have one) a small example that currently feels mysterious.

## Coming next

- [ ] Positional encodings / RoPE
- [ ] Autoregressive generation and sampling
- [ ] RMSNorm + SwiGLU
- [ ] KV cache
- [ ] Grouped query attention
