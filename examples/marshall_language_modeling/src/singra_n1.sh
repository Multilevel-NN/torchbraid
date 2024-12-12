#!/bin/bash
#SBATCH --job-name=lang-model
#SBATCH --time=48:00:00
#SBATCH --output=ml_n1.out
#SBATCH --error=ml_n1.err
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

BATCH_SIZE=8
EPOCHS=10
PDATA=0.005
rm timing_data_plots_1_32*

#nsys profile -o profile_report mpirun -n 1 python main.py --steps 32 --epochs=$EPOCHS --endepoch 1 --input_text openwebtext --percent-data=$PDATA --batch-size=$BATCH_SIZE --lp-max-levels "(1,2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 4 --model_dimension 16 --num_heads 4 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1 --context_window 512
mpirun -n 1 nsys profile -o profile_report_%q{OMPI_COMM_WORLD_RANK} python main.py --steps 32 --epochs=$EPOCHS --endepoch 1 --input_text openwebtext --percent-data=$PDATA --batch-size=$BATCH_SIZE --lp-max-levels "(1,2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 4 --model_dimension 16 --num_heads 4 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1 --context_window 512
#mpirun -n 1 python main.py --steps 32 --epochs=$EPOCHS --endepoch 1 --input_text openwebtext --percent-data=$PDATA --batch-size=$BATCH_SIZE --lp-max-levels "(1,2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 4 --model_dimension 16 --num_heads 4 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1 --context_window 512