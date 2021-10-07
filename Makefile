.PHONY: build rebuild clean clobber

UNAME_S := $(shell uname -s)

build:
	@echo $(uname)
	python setup.py build_ext -j2 --inplace

clean:
	rm -rf build dist wheelhouse
	rm -rf htmlcov .coverage .hypothesis
	rm -f nd2/*.so
	rm -f nd2/_*.c
	rm -f nd2/_sdk/*.so
	rm -f nd2/_sdk/*.c

clobber:
	make clean
	rm -rf sdk .mypy_cache

rebuild:
	make clean
	make build
