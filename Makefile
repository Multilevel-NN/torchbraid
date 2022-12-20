.PHONY: tests clean all 
.PHONY: test-direct-gpu  tests-direct-gpu tests test tests-serial test-serial

all:
	make -C ./src/torchbraid/test_fixtures
	make -C ./src/torchbraid

clean:
	make -C ./src/torchbraid/test_fixtures clean
	make -C ./src/torchbraid clean
	rm -fr examples/mnist/data
	rm -fr examples/cifar10/data

tests test:
	make -C ./tests tests

tests-serial test-serial:
	make -C ./tests tests-serial

tests-direct-gpu test-direct-gpu:
	make -C ./tests tests-direct-gpu
