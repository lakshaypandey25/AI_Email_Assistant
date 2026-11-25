.PHONY: install test lint run once auth mcp

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt

test:
	pytest --cov=email_assistant --cov-report=term-missing

lint:
	ruff check src tests && mypy src

auth:
	PYTHONPATH=src python -m email_assistant auth

once:
	PYTHONPATH=src python -m email_assistant run --once

run:
	PYTHONPATH=src python -m email_assistant run

mcp:
	PYTHONPATH=src python -m email_assistant mcp-server
