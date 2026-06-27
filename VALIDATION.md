# Validation

The test suite has 34 tests across 8 koan files.

I validated that the suite can be collected against the intended koan/TODO state across the focused modules in `src/llm_koans/`.

Expected state when you first run `pytest`:

- tests fail at TODOs
- each passing test confirms one concept or implementation step

Expected state after completing all TODOs:

- `34 passed`
