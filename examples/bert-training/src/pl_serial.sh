#!/bin/bash
#SBATCH -A m1327
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --time=00:25:00
#SBATCH -N 1 # Nodes
#SBATCH --ntasks-per-node=1
#SBATCh -c 128 # https://docs.nersc.gov/systems/perlmutter/running-jobs/#1-node-4-tasks-4-gpus-1-gpu-visible-to-each-task
#SBATCH --gpus-per-task=1 # Number of GPUs per MPI task; each rank has one
#SBATCH --gpu-bind=none
#SBATCH --output=ml_serial.out
#SBATCH --error=ml_serial.err


export HF_DATASETS_CACHE=$SCRATCH/huggingface_cache
export HF_HOME=$HF_DATASETS_CACHE

export SLURM_CPU_BIND="cores"
module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 python
conda activate gpu-aware-mpi
export MPICH_GPU_SUPPORT_ENABLED=1 

# -------------------- GENERATE TIMINGS ------------------------------ (not accuracy/long runs)
BATCH_SIZE=128
EPOCHS=2
PDATA=2500

# First generate the model; will auto leave with serial file
#srun -n 1 python main.py --serial-file True --percent-data=$PDATA --steps 32 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
#python main_serial.py --percent-data=$PDATA --steps 32 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
#rm serialnet_bert_32

# First generate the model; will auto leave with serial file
#srun -n 1 python main.py --serial-file True --percent-data=$PDATA --steps 64 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
#python main_serial.py --percent-data=$PDATA --steps 64 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
#rm serialnet_bert_64

# First generate the model; will auto leave with serial file
srun -n 1 python main.py --serial-file True --percent-data=$PDATA --steps 128 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
python main_serial.py --percent-data=$PDATA --steps 128 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
rm serialnet_bert_128

srun -n 1 python main.py --serial-file True --percent-data=$PDATA --steps 192 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
python main_serial.py --percent-data=$PDATA --steps 192 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
rm serialnet_bert_192

# ----------------------------------------------------------


# https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py
# "applications using mpi4py must be launched via srun"

#srun ./select_gpu_device python test-gpu-aware-mpi.py
#srun -n 2 -c 32 python  main.py --steps 12 --channels 8 --batch-size 50 --log-interval 100 --epochs 20
# srun python main.py
# srun --ntasks 4 --gpus-per-task 1 -c 32 --gpu-bind=none python main_noDP.py > 4.out
