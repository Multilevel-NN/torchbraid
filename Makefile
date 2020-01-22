MODULE_NAME=torchbraid

all: $(MODULE_NAME).cpython-37m-darwin.so 

braid.o: braid.c
	mpicc -fPIC -c braid.c

libbraid.a: braid.o
	ar cruv libbraid.a braid.o

$(MODULE_NAME).cpython-37m-darwin.so: $(MODULE_NAME).pyx 
	CC=mpicc python setup.py install

clean:
	rm -fr $(MODULE_NAME).c $(MODULE_NAME).h ${MODULE_NAME}_callbacks.c braid.o libbraid.a build $(MODULE_NAME).cpython-37m-darwin.so
