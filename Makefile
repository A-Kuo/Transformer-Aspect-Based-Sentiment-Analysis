.PHONY: install install-dev train evaluate predict test lint clean info help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -e .

install-dev:  ## Install with dev dependencies (lint, test, pre-commit)
	pip install -e ".[dev]"
	pre-commit install

train:  ## Train the model with default config
	python main.py train

evaluate:  ## Evaluate the best checkpoint on the test set
	python main.py evaluate

predict:  ## Interactive prediction mode
	python main.py predict

info:  ## Print config and architecture summary
	python main.py info

test:  ## Run the smoke test suite
	pytest

test-quick:  ## Run tests that skip BERT download
	pytest -k "TestConfig or TestEvaluationMetrics or test_aspect_keywords or test_assign_aspect or test_stars_to_sentiment"

lint:  ## Lint with ruff
	ruff check src/ tests/ main.py

lint-fix:  ## Auto-fix lint issues
	ruff check --fix src/ tests/ main.py

clean:  ## Remove generated artifacts
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .ruff_cache
	rm -rf dist build *.egg-info
	rm -rf models/*.pt results/*.json
