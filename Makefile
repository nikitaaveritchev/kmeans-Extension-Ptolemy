TEX_DEPS := paper.tex content/* fig/* *.tex references/*
TEX_OPTIONS := --shell-escape -output-directory=build -interaction=nonstopmode -halt-on-error

paper.pdf: $(TEX_DEPS)
	mkdir -p build 
	lualatex $(TEX_OPTIONS) paper.tex
	bibtex build/paper 
	lualatex $(TEX_OPTIONS) paper.tex
	lualatex $(TEX_OPTIONS) paper.tex
	qpdf --linearize --newline-before-endstream build/paper.pdf paper.pdf

all: paper.pdf

.PHONY: clean
clean:
	rm -r build
	rm -r paper.pdf
