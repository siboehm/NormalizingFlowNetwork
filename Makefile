test:
	python -m pytest -v

black: 
	black --exclude venv .
