.PHONY: test koan install

install:
	python -m pip install -e .

test:
	python -m pytest

# Run a single koan file, for example: make koan N=04
koan:
	python -m pytest tests/test_$(N)*.py -q
