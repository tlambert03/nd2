.PHONY: rebuild clean build

build:
	python setup.py build_ext -j2 --inplace

clean:
	rm -rf build dist wheelhouse
	rm -rf htmlcov .coverage .hypothesis
	rm -f nd2/*.so
	rm -f nd2/*.c

rebuild:
	make clean
	make build
