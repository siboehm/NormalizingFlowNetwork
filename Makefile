.PHONY: tests
tests:
	python -m pytest -v

quicktest:
	python -m pytest -v -m "not slow"

black: 
	black --exclude "(venv|\.json)" --line-length 100 .

itecdata:
	scp -r "boehm@i61pc004.itec.uni-karlsruhe.de:/common/homes/students/boehm/Documents/VI/data/local/*" data/cluster

viz: itecdata
	python plot.py
