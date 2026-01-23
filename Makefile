.PHONY: lint lint-fix format type check

TARGET ?= ./src

lint:
	uv run ruff check $(TARGET)

lint-fix:
	uv run ruff check $(TARGET) --fix

format:
	uv run ruff format $(TARGET)

type:
	uv run mypy $(TARGET)

fix:
	uv run ruff format $(TARGET)
	uv run ruff check $(TARGET) --fix --unsafe-fixes
	
check:
	uv run ruff format $(TARGET) --check
	uv run ruff check $(TARGET)
	uv run mypy $(TARGET)