include ../makefile.inc

all:

tests test:
	$(MPIRUN) -n 1 $(PYTHON) test_callbacks.py
	$(MPIRUN) -n 1 $(PYTHON) test_FlatPackUnpack.py
	$(MPIRUN) -n 3 $(PYTHON) test_layer_parallel.py
	$(MPIRUN) -n 3 $(PYTHON) test_layer_parallel_multinode.py
	$(MPIRUN) -n 3 $(PYTHON) test_composite.py
	$(MPIRUN) -n 3 $(PYTHON) test_grad_update.py
	$(MPIRUN) -n 3 $(PYTHON) test_gru_layer_parallel_new.py
	$(PYTHON) test_ContextTimer.py
	$(PYTHON) test_mean_initial_guess.py

tests-serial test-serial:
	$(MPIRUN) -n 1 $(PYTHON) test_callbacks.py
	$(MPIRUN) -n 1 $(PYTHON) test_FlatPackUnpack.py
	$(MPIRUN) -n 1 $(PYTHON) test_layer_parallel.py
	$(MPIRUN) -n 1 $(PYTHON) test_layer_parallel_multinode.py
	$(MPIRUN) -n 1 $(PYTHON) test_composite.py
	$(MPIRUN) -n 1 $(PYTHON) test_grad_update.py
	$(MPIRUN) -n 1 $(PYTHON) test_gru_layer_parallel_new.py
	$(PYTHON) test_ContextTimer.py
	$(PYTHON) test_mean_initial_guess.py

tests-direct-gpu test-direct-gpu:
	$(MPIRUN) -n 2 $(PYTHON) test_gpu_direct_commu.py
