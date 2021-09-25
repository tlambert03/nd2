.PHONY: rebuild clean build

UNAME_S := $(shell uname -s)

build:
	@echo $(uname)
	python setup.py build_ext -j2 --inplace
ifeq ($(UNAME_S),Darwin)
	@echo LINKING
	install_name_tool -change libnd2sdk.dylib @loader_path/../sdk_legacy/Darwin/shared/lib/libnd2sdk.dylib nd2/_nd2file_legacy.cpython-39-darwin.so
else
	@echo FAILED
endif

clean:
	rm -rf build dist wheelhouse
	rm -rf htmlcov .coverage .hypothesis
	rm -f nd2/*.so
	rm -f nd2/_*.c

rebuild:
	make clean
	make build
