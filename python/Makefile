MODULE_NAME=torchbraid

LIB_NAME=$(MODULE_NAME).cpython-37m-darwin.so
LIB_NAME=$(MODULE_NAME).cpython-36m-x86_64-linux-gnu.so

all: $(LIB_NAME)

$(LIB_NAME): $(MODULE_NAME).pyx 
	CC=mpicc python setup.py build_ext --inplace

clean:
	rm -fr $(MODULE_NAME).c $(MODULE_NAME).h ${MODULE_NAME}_callbacks.c build braid.out.cycle $(LIB_NAME)
