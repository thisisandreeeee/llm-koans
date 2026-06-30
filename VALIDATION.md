# Validation

The test suite now has 51 tests across 13 koan files.

I validated that the suite can be collected against the intended learner-facing TODO state across the focused modules in `src/llm_koans/`.

Expected state when you first run `pytest`:

- tests fail at TODOs
- each passing test confirms one concept or implementation step

For the new fine-tuning koans (`08`–`12`), I also validated both sides of the scaffold:

- learner-facing run: `17 failed` at `KoanIncomplete` TODOs
- temporary solved copy: `17 passed`

Expected state after completing all TODOs:

- `51 passed`
