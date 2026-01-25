.PHONY: lint lint-fix format type check

TARGET ?= ./src

lint:
	ruff check $(TARGET)

lint-fix:
	ruff check $(TARGET) --fix

format:
	ruff format $(TARGET)

type:
	mypy --ignore-missing-imports --disable-error-code=import-untyped $(TARGET)

fix:
	ruff format $(TARGET)
	ruff check $(TARGET) --fix --unsafe-fixes
	
check:
	ruff format $(TARGET) --check
	ruff check $(TARGET)
	mypy --ignore-missing-imports --disable-error-code=import-untyped $(TARGET)