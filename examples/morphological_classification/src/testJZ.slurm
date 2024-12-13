#!/bin/bash
#SBATCH --job-name=gpu_multi_mpi      # job name
# It is possible to use a partition other than the default
# by activating one of the following 5 directives:
##SBATCH -C v100-32g # uncomment to reserve only 16 GB V100 GPUs
#SBATCH --partition=gpu_p2
# Here, reservation of 3x10=30 CPUs (for 3 tasks) and 3 GPUs (1 GPU per task) on a single node:
#SBATCH --nodes=1                     # number of nodes
#SBATCH --ntasks-per-node=1           # number of MPI tasks per node (= number of GPUs per node)
#SBATCH --gres=gpu:1                  # number of GPUs per node (max 8 with gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=2            # number of CPUs per task (1/4 of the 4-GPU V100 node here)
#SBATCH --hint=nomultithread          # hyperthreading disabled
#SBATCH --time=00:05:00               # maximum execution time requested (HH:MM:SS)
#SBATCH --output=MC_128_tmpi%j.out  # output file name
#SBATCH --error=MC_128_tmpi%j.out   # error file name (here common with output)


# Cleaning modules loaded in interactive mode and inherited by default
module purge


# Loading modules
module load pytorch-gpu/py3/2.2.0
export PYTHONUSERBASE=/lustre/fswork/projects/rech/emb/ump43xm/torchbraid/torchbraid

# Echo of commands issued
set -x


srun  python -u main_noDP_NI_callibration.py --enforce_serial  --batch-size 4 --epochs 1 --optimizer SGD --lr 1e-2 --momentum .9 --gradient_accumulation 1 --model_dimension 128 --num_heads 1 --steps 128 --Tf 128  --lp-cfactor 8 --lp-max-levels 2 --lp-fwd-max-iters 2 --lp-bwd-max-iters 1 --seed 0 --scheduler None --ni_cfactor 2 --ni_num_levels 1 --ni_interpolation linear --ni_interpolate_momentum True


