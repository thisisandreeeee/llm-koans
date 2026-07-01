# Validation

The test suite now has 57 tests across 13 koan files.

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

Expected state after completing all TODOs:

- `57 passed`
