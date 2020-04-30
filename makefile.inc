MPIRUN=mpirun
PYTHON=python

# For my Linux
#####################################
XBRAID_ROOT=/home/eccyr/Packages/xbraid/braid
POSTFIX=cpython-36m-x86_64-linux-gnu.so  
CC=mpicc
LDSHARED=mpicc -shared
EXTRA_FLAGS=-O3 -fPIC

# for my MAC
#####################################
# POSTFIX=cpython-36m-darwin.so
# XBRAID_ROOT=/Users/eccyr/Packages/xbraid/braid
# CC=mpicc
# LDSHARED=
# EXTRA_FLAGS=
