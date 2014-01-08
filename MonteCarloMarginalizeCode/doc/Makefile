
CODEDIR=../Code/
DOCFILES=${CODEDIR}dag_utils.py ${CODEDIR}lalsimutils.py ${CODEDIR}statutils.py ${CODEDIR}xmlutils.py ${CODEDIR}factored_likelihood.py ${CODEDIR}mcsampler.py

index.html: ${DOCFILES}
	epydoc-2.7 -v -o ./ --html --name "mcparamest" --top "index.html" ${DOCFILES}

clean:
	rm -rf *.html
