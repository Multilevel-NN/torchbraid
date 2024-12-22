#!/bin/bash
#SBATCH --job-name=lang-model-2
#SBATCH --time=80:30:00
#SBATCH --output=ml_multi_2.out
#SBATCH --error=ml_multi_2.err
#SBATCH --nodelist=sn[5-16]

#SBATCH --nodes=2
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

#mpirun -n 2 python main.py --percent-data .01 --steps 32 --epochs 3 --batch-size 32 --lp-max-levels 2 --lp-bwd-max-iters 1 --lp-fwd-max-iters 3 --lp-cfactor 4 --model_dimension 384 --num_heads 6 --lp-print-level 1 --lp-braid-print-level 1 --Tf 1

# -------------- GENERATE TIMINGS --------------- (not accuracy long runs)
#BATCH_SIZE=256
#EPOCHS=2
#PDATA=4000
#mpirun -n 2 python main.py --percent-data=$PDATA --steps 32 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels 2 --lp-bwd-max-iters 1 --lp-fwd-max-iters 3 --lp-cfactor 4 --model_dimension 384 --num_heads 6 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1

#mpirun -n 2 python main.py --percent-data=$PDATA --steps 64 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels 2 --lp-bwd-max-iters 1 --lp-fwd-max-iters 3 --lp-cfactor 4 --model_dimension 384 --num_heads 6 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1

#mpirun -n 2 python main.py --percent-data=$PDATA --steps 128 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels 2 --lp-bwd-max-iters 1 --lp-fwd-max-iters 3 --lp-cfactor 4 --model_dimension 384 --num_heads 6 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1


#mpirun -n 2 python main.py --percent-data=$PDATA --steps 128 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(1,2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 3 --lp-cfactor 4 --model_dimension 384 --num_heads 6 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1

#------------------------------------------------------------------------


# --------------- GENERATE ACCURACY  -----
BATCH_SIZE=256
EPOCHS=1
#PDATA=10000
PDATA=5000

#mpirun -n 2 python main.py --percent-data=$PDATA --steps 64 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(1, 2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 4 --model_dimension 384 --num_heads 6 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1

#mpirun -n 2 python main.py --percent-data=$PDATA --steps 64 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(1, 3)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 4 --model_dimension 384 --num_heads 6 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1

#mpirun -n 2 python main.py --percent-data=$PDATA --steps 128 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(1, 2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 4 --model_dimension 384 --num_heads 6 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1

#mpirun -n 2 python main.py --percent-data=$PDATA --steps 128 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(1, 3)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 4 --model_dimension 384 --num_heads 6 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1


# ------------------------------------

# Long run to see if timings are bad.... 

BATCH_SIZE=256
EPOCHS=1
PDATA=300000  #20000

#mpirun -n 2 python main.py --percent-data=$PDATA --steps 32 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(1, 2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 4 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1 --log-interval 1
#mpirun -n 2 python main.py --percent-data=$PDATA --steps 32 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(2, 2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 2 --lp-cfactor 4 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1

mpirun -n 2 python main.py --percent-data=$PDATA --steps 64 --epochs=$EPOCHS --batch-size=$BATCH_SIZE --lp-max-levels "(1, 2)" --lp-bwd-max-iters 1 --lp-fwd-max-iters 1 --lp-cfactor 4 --lp-print-level 0 --lp-braid-print-level 0 --Tf 1 --log-interval 1

