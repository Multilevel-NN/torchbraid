#!/bin/bash
#SBATCH -A m1327
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --time=00:10:00
#SBATCH -N 2 # Nodes
#SBATCH --ntasks-per-node=4
#SBATCh -c 32 # https://docs.nersc.gov/systems/perlmutter/running-jobs/#1-node-4-tasks-4-gpus-1-gpu-visible-to-each-task
#SBATCH --gpus-per-task=1 # Number of GPUs per MPI task; each rank has one
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND="cores"

module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 python
conda activate gpu-aware-mpi
export MPICH_GPU_SUPPORT_ENABLED=1 

# https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py
# "applications using mpi4py must be launched via srun"

#srun ./select_gpu_device python test-gpu-aware-mpi.py
#srun -n 2 -c 32 python  main.py --steps 12 --channels 8 --batch-size 50 --log-interval 100 --epochs 20
srun python  main.py
# srun --ntasks 4 --gpus-per-task 1 -c 32 --gpu-bind=none python main_noDP.py > 4.out
