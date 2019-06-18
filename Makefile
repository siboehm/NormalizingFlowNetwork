.PHONY: tests data
ITEC_URL = "i61pc004.itec.uni-karlsruhe.de"
ITEC_VI_PATH= "/common/homes/students/boehm/Documents/VI"
HPC_URL = "bwunicluster.scc.kit.edu"
HPC_VI_PATH="/home/kit/fbv/gd5482/sboehm/VI"

tests:
	python -m pytest -v

quicktest:
	python -m pytest -v -m "not slow"

black: 
	black --exclude "(venv|\.json)" --line-length 100 .

itecdata:
	scp -r "boehm@$(ITEC_URL):$(ITEC_VI_PATH)/data/local/*" data/cluster

hpcdata:
	scp -r "gd5482@$(HPC_URL):$(HPC_VI_PATH)/data/local/*" data/cluster

data: hpcdata

viz: data
	python plot.py
