.PHONY: tests clean uninstall all

all: 
	make -C ./torchbraid
	make -C ./tests

clean:
	make -C ./torchbraid clean
	make -C ./tests clean

uninstall:
	make -C ./torchbraid uninstall
	make -C ./tests uninstall

tests:
	make -C ./tests tests
