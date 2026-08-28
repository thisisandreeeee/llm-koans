# LLM Koans

A hands-on set of practical Python koans for building intuition about attention, Transformers, and LLM training mechanics.

The koans teach the core mechanics behind modern LLMs:

- **PyTorch matmul rule** = last two dims are the matrix; earlier dims are batch.
- **Query** = what this token is looking for.
- **Key** = how another token advertises that it is relevant.
- **Value** = the information retrieved if that token is attended to.
- **Dot product** = an alignment/similarity score between a query and a key.
- **Softmax** = converts raw scores into positive weights that sum to 1.
- **Context vector** = a weighted blend of values; the token after gathering context.
- **Encoder/decoder blocks** = the classic Transformer building blocks.
- **Mixture of Experts (MoE)** = replace the dense FFN sublayer with a router and multiple token-local FFN experts.
- **Transformer assembly** = compose nn.Embedding, nn.TransformerEncoder/Decoder, and task heads into GPT‑style, BERT‑style, and T5‑style architectures from the same building blocks.
- **Training loop** = shift logits for next-token prediction, compute loss, backprop, step.
- **SFT data formatting** = apply a chat template and mask prompt tokens out of the loss.
- **Fine-tuning** = adapt a base model by choosing which parameters update and which loss to optimize.
- **LoRA/PEFT lifecycle** = train, save/load, and merge small low-rank adapters over a frozen base.
- **Distillation** = train a smaller student from a teacher's softened output distribution.
- **Preference tuning** = use chosen-vs-rejected examples to train DPO-style policy updates.

The attention exercises are inspired by the step-by-step structure in Sebastian Raschka's article, "Understanding and Coding the Self-Attention Mechanism of Large Language Models From Scratch":

https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html

The training exercises are informed by *LLMs in Production* by Christopher Brousseau and Matthew Sharp, especially the chapter 5 sections on fine-tuning, distillation, RLHF, LoRA/PEFT, and the adaptation tradeoff between full updates and parameter-efficient updates.

## Repository layout

```text
llm-koans/
├── src/llm_koans/                 # You edit the focused koan modules here
│   ├── koan_00_shapes_and_projections.py
│   ├── koan_01_attention_scores.py
│   ├── koan_02_self_attention.py
│   ├── koan_03_multihead_attention.py
│   ├── koan_04_masks.py
│   ├── koan_05_blocks.py
│   ├── koan_06_moe.py
│   ├── koan_07_training.py
│   ├── koan_08_finetuning.py
│   ├── koan_09_lora_lifecycle.py
│   ├── koan_10_distillation.py
│   ├── koan_11_dpo.py
│   ├── common.py                  # Shared helpers, no koan prefix
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
pytest tests/test_00_shapes_and_projections.py -q
pytest tests/test_01_attention_scores.py -q
pytest tests/test_02_self_attention.py -q
pytest tests/test_03_multihead_attention.py -q
pytest tests/test_04_masks_and_decoder_attention.py -q
pytest tests/test_05_encoder_decoder_blocks.py -q
pytest tests/test_06_moe.py -q
pytest tests/test_07_training_updates.py -q
pytest tests/test_08_finetuning.py -q
pytest tests/test_09_lora_lifecycle.py -q
pytest tests/test_10_distillation.py -q
pytest tests/test_11_dpo.py -q
```

Or use the helper:

```bash
python tools/check.py
python tools/check.py 03
```

## Suggested learning loop

1. Open the failing test.
2. Read the test name and comments.
3. Implement only the function needed for that test in the matching focused module.
4. Run the test again.
5. Move to the next test file.

This is intentionally not a polished library. It is a learning repo. The tests are the teacher.

## PyTorch matmul rule of thumb

For tensors with two or more dimensions, `torch.matmul` treats the final two
axes as the matrix and every earlier axis as batch:

```text
(..., rows, shared) @ (..., shared, cols) -> (..., rows, cols)
```

The leading batch axes must either match or be broadcastable. This is why:

```text
(B, T, D) @ (D, E)              -> (B, T, E)
(B, H, Tq, D) @ (B, H, D, Tk)   -> (B, H, Tq, Tk)
```

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
