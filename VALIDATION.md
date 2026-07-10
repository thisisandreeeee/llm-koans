# Validation

The test suite now has 81 tests across 15 koan files.

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

For the encoder/decoder block koan (`06`):

- learner-facing run: `4 failed` at `KoanIncomplete` TODOs
- temporary solved copy: `4 passed`

For the mixture-of-experts koan (`13`):

- builds on the block mechanics from Koan 06: MoE replaces the position-wise FFN sublayer
- learner-facing run: `4 failed` at `KoanIncomplete` TODOs
- temporary solved run: `4 passed`

For the tool-calling koan (`14`):

- teaches the minimal function-calling chatbot loop: schema -> assistant tool call -> Python dispatch -> tool message -> final assistant answer
- learner-facing run: `5 failed` at `KoanIncomplete` TODOs
- temporary solved run: `5 passed`

Expected state after completing all TODOs:

- `81 passed`
