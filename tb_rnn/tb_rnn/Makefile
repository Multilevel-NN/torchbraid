include ../makefile.inc

MODULE_NAME=torchbraid_rnn

LIB_NAME=$(MODULE_NAME)_app.$(POSTFIX)

all: $(LIB_NAME) 

$(LIB_NAME): ${MODULE_NAME}_app.pyx ${MODULE_NAME}_callbacks.pyx
	#XBRAID_ROOT=$(XBRAID_ROOT) CC=$(CC) $(PYTHON) setup.py install --record installed_files.txt
	XBRAID_ROOT=$(XBRAID_ROOT) CC=$(CC) $(PYTHON) setup.py build_ext --inplace

clean:
	rm -fr $(MODULE_NAME)_app.c $(MODULE_NAME)_app.h ${MODULE_NAME}_callbacks.c build braid.out.cycle *.html __pycache__/ $(LIB_NAME)

uninstall:
	#cat installed_files.txt | xargs rm -fr # remove installation
	#rm -fr installed_files.txt	
	echo "uninstall - no-op"
