#!/usr/bin/env bash

set -e

DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 || (echo "Could not change into current directory!" && exit 1) ; pwd -P )"

mpiexec --quiet -n 1 --use-hwthread-cpus --oversubscribe python3 -m mpi4py ${DIR}/../$1.py
#mpiexec --quiet -n 2 --use-hwthread-cpus --oversubscribe python3 -m mpi4py ${DIR}/../$1.py
mpiexec --quiet -n 3 --use-hwthread-cpus --oversubscribe python3 -m mpi4py ${DIR}/../$1.py
