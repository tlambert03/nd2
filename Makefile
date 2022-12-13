.PHONY: build rebuild clean clobber

UNAME_S := $(shell uname -s)

build:
	@echo $(uname)
	python setup.py build_ext -j2 --inplace

clean:
	rm -rf build dist wheelhouse
	rm -rf htmlcov .coverage .hypothesis
	rm -f src/nd2/*.so
	rm -f src/nd2/_*.c
	rm -f src/nd2/_sdk/*.so
	rm -f src/nd2/_sdk/*.c
	rm -f src/nd2/_sdk/*.cpp
	rm -f src/nd2/_pysdk/*.so

clobber:
	make clean
	rm -rf .mypy_cache

rebuild:
	make clean
	make build
