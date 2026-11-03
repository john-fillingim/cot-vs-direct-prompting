# Pipeline targets. On Windows without make, use .\run_all.ps1 instead.
PY ?= python

.PHONY: all pilot data clean analyze figures test lint

all: data clean analyze figures

pilot:
	$(PY) -m src.sample_problems
	$(PY) -m src.run_experiment --pilot
	$(PY) -m src.clean --pilot
	$(PY) -m src.power_analysis --pilot

data:
	$(PY) -m src.sample_problems
	$(PY) -m src.run_experiment

clean:
	$(PY) -m src.clean

analyze:
	$(PY) -m src.analyze

figures:
	$(PY) -m src.figures

test:
	$(PY) -m pytest -q

lint:
	ruff check .
