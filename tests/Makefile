include ../makefile.inc

MODULE_NAME=torchbraid_test

LIB_NAME=$(MODULE_NAME).$(POSTFIX)

all: $(LIB_NAME)

$(LIB_NAME): ${MODULE_NAME}.pyx
	#XBRAID_ROOT=$(XBRAID_ROOT) CC=mpicc $(PYTHON) setup.py install --record installed_files.txt
	XBRAID_ROOT=$(XBRAID_ROOT) CC=$(CC) $(PYTHON) setup.py build_ext --inplace

clean:
	rm -fr $(MODULE_NAME).c $(MODULE_NAME).h build *.html __pycache__/ $(LIB_NAME)

uninstall:
	cat installed_files.txt | xargs rm -fr # remove installation
	rm -fr installed_files.txt

tests test:
	$(MPIRUN) -n 3 $(PYTHON) test_layer_parallel.py
	$(MPIRUN) -n 3 $(PYTHON) test_composite.py
	$(MPIRUN) -n 3 $(PYTHON) test_grad_update.py
	$(MPIRUN) -n 2 $(PYTHON) test_rnn_layer_parallel.py
	$(PYTHON) test_ContextTimer.py

tests-serial test-serial:
	$(MPIRUN) -n 1 $(PYTHON) test_layer_parallel.py
	$(MPIRUN) -n 1 $(PYTHON) test_composite.py
	$(MPIRUN) -n 1 $(PYTHON) test_grad_update.py
	$(MPIRUN) -n 1 $(PYTHON) test_rnn_layer_parallel.py
	$(PYTHON) test_ContextTimer.py
