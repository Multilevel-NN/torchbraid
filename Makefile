.PHONY: tests clean uninstall all

all: 
	make -C ./torchbraid
	make -C ./tests

clean:
	make -C ./torchbraid clean
	make -C ./tests clean
	rm -fr examples/mnist/data
	rm -fr examples/cifar10/data

uninstall:
	make -C ./torchbraid uninstall
	make -C ./tests uninstall

tests test:
	make -C ./tests tests
