.PHONY: tests
tests:
	python -m pytest -v

quicktest:
	python -m pytest -v -m "not slow"

black: 
	black --exclude "(venv|\.json)" .
