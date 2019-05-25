.PHONY: tests
tests:
	python -m pytest -v

black: 
	black --exclude venv .
