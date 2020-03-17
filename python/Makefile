MODULE_NAME=torchbraid

LIB_NAME=$(MODULE_NAME).cpython-37m-darwin.so            # for MAC
LIB_NAME=$(MODULE_NAME).cpython-36m-darwin.so            # for MAC
#LIB_NAME=$(MODULE_NAME).cpython-36m-x86_64-linux-gnu.so  # for linux

all: $(LIB_NAME)

$(LIB_NAME): $(MODULE_NAME).pyx ${MODULE_NAME}_callbacks.pyx
	#CC=mpicc python setup.py build_ext --inplace
	CC=mpicc python setup.py install

clean:
	rm -fr $(MODULE_NAME).c $(MODULE_NAME).h ${MODULE_NAME}_callbacks.c build braid.out.cycle *.html __pycache__/ $(LIB_NAME)
