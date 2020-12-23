.PHONY:example 
example:
	mpirun -n 2 python forward_scaling_RNN_MNIST.py          2 | tail -n 20
	echo "=============================================="
	mpirun -n 1 python forward_scaling_RNN_MNIST.py          2 | tail -n 18
	echo "=============================================="
	mpirun -n 1 python forward_scaling_RNN_MNIST.py --serial 2 | tail -n 32
