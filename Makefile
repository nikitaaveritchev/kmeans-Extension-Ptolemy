all: paper.pdf
code: results/combined_plot.pdf


TEX_DEPS := paper/paper.tex paper/content/* paper/*.tex paper/references/*
TEX_OPTIONS := --shell-escape -output-directory=build -interaction=nonstopmode -halt-on-error

paper.pdf paper/fig/combined_plot.pdf: $(TEX_DEPS) results/combined_plot.pdf check-dependencies
	mkdir -p paper/fig
	cp results/combined_plot.pdf paper/fig/combined_plot.pdf
	mkdir -p paper/build 
	cd paper && lualatex $(TEX_OPTIONS) paper.tex
	cd paper && bibtex build/paper 
	cd paper && lualatex $(TEX_OPTIONS) paper.tex
	cd paper && lualatex $(TEX_OPTIONS) paper.tex
	qpdf --linearize --newline-before-endstream paper/build/paper.pdf paper.pdf || cp paper/build/paper.pdf paper.pdf
	
results/combined_plot.pdf results/clustering_performace_results_formatted.xlsx: main.py means.py .venv
	mkdir -p results
	.venv/bin/python main.py

.venv: requirements.txt check-dependecies
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt


check-dependencies:
	@if ! command -v python3 &> /dev/null; then \
		echo "Python is not installed. Please install Python."; \
		exit 1; \
	fi
	@if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"; then \
		echo "Python version is very old. This software was not tested on your python version."; \
	fi
	@if ! command -v lualatex &> /dev/null; then \
		echo "lualatex is not installed. Please install lualatex."; \
		exit 1; \
	fi
clean:
	-rm -r results
	-rm -r paper/build
	-rm -r .venv
	-rm -r paper/fig/combined_plot.pdf
	
.PHONY: clean all code

