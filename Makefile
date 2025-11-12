.PHONY: setup test rf sim plots clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  make setup    - Install dependencies"
	@echo "  make test     - Run smoke test"
	@echo "  make rf       - Run RF training + evaluation"
	@echo "  make sim      - Run macro scenario MC simulation"
	@echo "  make plots    - Generate plots from latest results"
	@echo "  make clean    - Clean cache and temporary files"

# Install dependencies
setup:
	pip install -r requirements.txt

# Run smoke test
test:
	@echo "Running smoke test..."
	@python tests/smoke_test.py || python -m pytest tests/ -q

# Run Random Forest training and evaluation
rf:
	python -m finmc_tech.cli train-rf

# Run macro scenario Monte Carlo simulation
sim:
	python -m finmc_tech.cli simulate --shock base --h 24 --n 200

# Generate plots from latest results
plots:
	python -m finmc_tech.cli plots --which all

# Clean cache and temporary files
clean:
	rm -rf data_cache/*
	rm -rf results/*.json
	rm -rf results/*.png
	rm -rf __pycache__/
	rm -rf finmc_tech/**/__pycache__/
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

