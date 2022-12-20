.PHONY: clean libs

include ../../makefile.inc

MODULE_NAME=torchbraid

all: libs utils

libs: ${MODULE_NAME}_app.pyx ${MODULE_NAME}_callbacks.pyx
	XBRAID_ROOT=$(XBRAID_ROOT) CC=$(CC) $(PYTHON) setup.py build_ext --inplace

utils:
	make -C ./utils

clean:
	rm -fr *.c *.h build braid.out.cycle *.html __pycache__/ *.so
	make -C ./utils         clean 
	make -C ./test_fixtures clean 
