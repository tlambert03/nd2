.PHONY: rebuild clean build

UNAME_S := $(shell uname -s)

build:
	@echo $(uname)
	python setup.py build_ext -j2 --inplace
ifeq ($(UNAME_S),Darwin)
	install_name_tool -change libnd2ReadSDK.dylib @loader_path/../sdk/v9/Darwin/lib/libnd2ReadSDK.dylib nd2/_nd2file_legacy.cpython-39-darwin.so
endif

clean:
	rm -rf build dist wheelhouse
	rm -rf htmlcov .coverage .hypothesis
	rm -f nd2/*.so
	rm -f nd2/_*.c

rebuild:
	make clean
	make build
