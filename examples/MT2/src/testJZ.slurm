#!/bin/bash
#SBATCH --job-name=gpu_multi_mpi      # job name
# It is possible to use a partition other than the default
# by activating one of the following 5 directives:
##SBATCH -C v100-32g # uncomment to reserve only 16 GB V100 GPUs
#SBATCH --partition=gpu_p2
# Here, reservation of 3x10=30 CPUs (for 3 tasks) and 3 GPUs (1 GPU per task) on a single node:
#SBATCH --nodes=1                     # number of nodes
#SBATCH --ntasks-per-node=2           # number of MPI tasks per node (= number of GPUs per node)
#SBATCH --gres=gpu:2                  # number of GPUs per node (max 8 with gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=2             # number of CPUs per task (1/4 of the 4-GPU V100 node here)
#SBATCH --hint=nomultithread          # hyperthreading disabled
#SBATCH --time=20:00:00               # maximum execution time requested (HH:MM:SS)
#SBATCH --output=long_term%j.out  # output file name
#SBATCH --error=long_term%j.out   # error file name (here common with output)


# Cleaning modules loaded in interactive mode and inherited by default
module purge


# Loading modules
module load pytorch-gpu/py3/2.2.0
export PYTHONUSERBASE=/lustre/fswork/projects/rech/emb/ump43xm/torchbraid/torchbraid

# Echo of commands issued
set -x


srun python3 -u main.py --batch-size 8 --epochs 1000000 --d_model 512 --dropout .1 --gradient_accumulation 16 --initialize_parameters --num_warmup_steps 8000 --tokenization unigram --vocab_size 8000 --steps 6 --Tf 6. --lp-max-levels 2 --lp-cfactor 3 --lp-fwd-max-iters 3 --lp-bwd-max-iters 2 --seed 0 --num_training_batches 20000 --load



