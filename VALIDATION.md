# Validation

The test suite now has 76 tests across 13 koan files.

I validated that the suite can be collected against the intended learner-facing TODO state across the focused modules in `src/llm_koans/`.

Expected state when you first run `pytest`:

- tests fail at TODOs
- each passing test confirms one concept or implementation step

For the transformer training koan (`07`):

- learner-facing run: `9 failed` at `KoanIncomplete` TODOs
- temporary solved copy: `9 passed`
- exercises: causal LM (GPT‑style), bidirectional encoder (BERT‑style), encoder‑decoder (T5‑style)

For the fine-tuning koans (`08`–`12`):

- learner-facing run: `17 failed` at `KoanIncomplete` TODOs
- temporary solved copy: `17 passed`

For the encoder/decoder block koan (`06`), now extended with MoE:

- anchor: Koan 06 is the most suitable concept to extend because MoE replaces the position-wise FFN sublayer inside an attention block, rather than changing the attention mechanism itself
- learner-facing run: `8 failed` at `KoanIncomplete` TODOs
- temporary solved run: `8 passed`
- added exercises: router logits, top-1 expert routing, routed expert FFN, MoE encoder block

Expected state after completing all TODOs:

- `76 passed`
