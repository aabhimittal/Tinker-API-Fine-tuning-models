.PHONY: help install dev test lint fmt typecheck serve docker clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

dev:  ## Install with dev + tinker extras
	pip install -e ".[dev]"

test:  ## Run the test suite
	pytest -q

lint:  ## Lint with ruff
	ruff check src tests

fmt:  ## Auto-format / auto-fix with ruff
	ruff check --fix src tests

typecheck:  ## Static type-check with mypy
	mypy src

serve:  ## Run the API locally (dry-run)
	uvicorn tinker_finetune.api.app:app --reload --port 8000

docker:  ## Build the Docker image
	docker build -f docker/Dockerfile -t tinker-finetune:latest .

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
