#!/bin/bash
#SBATCH --job-name=lang-model-serial
#SBATCH --time=20:30:00
#SBATCH --output=ml_serial.out
#SBATCH --error=ml_serial.err
#SBATCH --nodelist=sn[5-16]

#SBATCH --nodes=1
#SBATCH --partition=ampere
#SBATCH --ntasks-per-node=1

module purge

module load gcc/12.2.0
module load openmpi/4.1.4
module load cuda/11.8.0   
module load cudnn/8.4.0.27-11.6 
module load ucx/1.13.1 
module load python/3.10.8

source ~/braids/pip-test/bin/activate


# -------------------- GENERATE TIMINGS ------------------------------ (not accuracy/long runs)
#BATCH_SIZE=256
#EPOCHS=2
#PDATA=4000

# First generate the model; will auto leave with serial file
#mpirun -n 1 python main.py --serial-file True --percent-data=$PDATA --steps 32 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
#python main_serial.py --percent-data=$PDATA --steps 32 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
#rm serialnet_bert_32

# First generate the model; will auto leave with serial file
#mpirun -n 1 python main.py --serial-file True --percent-data=$PDATA --steps 64 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
#python main_serial.py --percent-data=$PDATA --steps 64 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
#rm serialnet_bert_64

# First generate the model; will auto leave with serial file
#mpirun -n 1 python main.py --serial-file True --percent-data=$PDATA --steps 128 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
#python main_serial.py --percent-data=$PDATA --steps 128 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
#rm serialnet_bert_128


# ----------------------------------------------------------

# -------------------- GENERATE ACCURACY ------------------------------ 
BATCH_SIZE=256
EPOCHS=3
PDATA=100000

# First generate the model; will auto leave with serial file
mpirun -n 1 python main.py --serial-file True --percent-data=$PDATA --steps 64 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
python main_serial.py --percent-data=$PDATA --steps 64 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --model_dimension 384 --num_heads 6
rm serialnet_bert_64

# ----------------------------------------------------------


# mpirun -n 2 python main.py

# Following is runs for getting losses for comparisons
# Baseline (no iterations)
# python main.py --epochs 10 --lp-max-levels 1 --num_heads 1 --steps 32

# Run different ones
# python main.py --epochs 10 --lp-max-levels 2 --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --num_heads 1 --steps 32
# python main.py --epochs 10 --lp-max-levels 2 --lp-bwd-max-iters 2 --lp-fwd-max-iters 2 --num_heads 1 --steps 32

# python main.py --epochs 10 --lp-max-levels 3 --lp-bwd-max-iters 2 --lp-fwd-max-iters 2 --num_heads 1 --steps 32
# python main.py --epochs 10 --lp-max-levels 3 --lp-bwd-max-iters 1 --lp-fwd-max-iters 2 --num_heads 1 --steps 32




