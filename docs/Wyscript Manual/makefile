TARGET = JavaScriptFileWriter_Manual
DEPS = introduction.tex types.tex functions.tex control-flow.tex

all: $(TARGET).pdf

%.pdf : %.tex $(DEPS)
	@for X in 1 2 3 ; do \
	( echo "---------------- Latex ($$X) ----------------" && pdflatex $< ) \
        done

clean:
	rm *.toc *.bbl *.aux *.dvi *.log *.ps *.pdf *~ *.blg *.glo *.gls *.glg *.idx *.xdy *.out *.ist *.bak
