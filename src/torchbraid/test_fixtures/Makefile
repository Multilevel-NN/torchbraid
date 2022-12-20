.PHONY: clean libs

include ../../../makefile.inc

MODULE_NAME=test_cbs

all: libs

libs: ${MODULE_NAME}.pyx ../torchbraid_callbacks.pyx
	XBRAID_ROOT=$(XBRAID_ROOT) CC=$(CC) $(PYTHON) setup.py build_ext --inplace

clean:
	rm -fr $(MODULE_NAME).c $(MODULE_NAME).h build *.html __pycache__/ *.so
