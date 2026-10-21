# Run from paper/. PYTHON should be an interpreter with the packages in
# ../requirements.txt plus analysis/requirements.txt.
PYTHON ?= python3
TEX     = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all results paper arxiv clean

all: results paper

results:
	cd .. && $(PYTHON) paper/analysis/rebuild_results.py

paper:
	$(TEX) main.tex > /dev/null
	bibtex main > /dev/null
	$(TEX) main.tex > /dev/null
	$(TEX) main.tex > /dev/null
	@grep -E "^!|undefined" main.log || echo "main.pdf built"

arxiv: paper
	tar czf arxiv_upload.tar.gz main.tex main.bbl cas-sc.cls cas-common.sty \
	    cas-model2-names.bst thumbnails figs/*.pdf figs/*.tex tables/*.tex
	@echo "arxiv_upload.tar.gz ready"

clean:
	rm -f main.aux main.log main.out main.blg main.spl main.fls main.fdb_latexmk
