#!/bin/bash -l
#SBATCH --job-name=MT2
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --nodelist=sn[5-16]
#SBATCH --partition=ampere
#SBATCH --ntasks-per-node=1
#SBATCH --output=mt2_n2.out
#SBATCH --error=mt2_n2.err

module purge

module load gcc/12.2.0
module load openmpi/4.1.4
module load cuda/11.8.0
module load cudnn/8.4.0.27-11.6
module load ucx/1.13.1
module load python/3.10.8

source ~/braids/pip-test/bin/activate

#cd $LOADPATH
#source load_modules.sh
#cd $RUNPATH

#srun python3 -u main.py --batch-size 8 --epochs 1000000 --d_model 512 --dropout .1 --gradient_accumulation 16 --initialize_parameters --num_warmup_steps 8000 --tokenization unigram --vocab_size 8000 --steps 6 --Tf 6. --lp-max-levels 2 --lp-cfactor 3 --lp-fwd-max-iters 3 --lp-bwd-max-iters 2 --seed 0 --num_training_batches 20000
mpirun -n 2 python main.py --batch-size 8 --epochs 100 --d_model 512 --dropout .1 --gradient_accumulation 16 --initialize_parameters --num_warmup_steps 8000 --tokenization unigram --vocab_size 8000 --steps 6 --Tf 6. --lp-max-levels 2 --lp-cfactor 3 --lp-fwd-max-iters 3 --lp-bwd-max-iters 2 --seed 0 --num_training_batches 20000
