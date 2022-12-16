.PHONY: tests clean uninstall all

all:
	make -C ./src/torchbraid/test_fixtures
	make -C ./src/torchbraid
	make -C ./tests

clean:
	make -C ./src/torchbraid/test_fixtures clean
	make -C ./src/torchbraid clean
	make -C ./tests clean
	rm -fr examples/mnist/data
	rm -fr examples/cifar10/data

uninstall:
	make -C ./src/torchbraid/test_fixtures uninstall
	make -C ./src/torchbraid uninstall
	make -C ./tests uninstall

tests test:
	make -C ./tests tests

tests-serial test-serial:
	make -C ./tests tests-serial

tests-direct-gpu test-direct-gpu:
	make -C ./tests tests-direct-gpu

example examples:
	make -C ./examples examples
