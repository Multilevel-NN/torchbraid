#!/bin/bash

# List of GPU counts you want to test
NUM_GPUS=(1 2 4 8)
NETDEPTHS=(32 64 128)

for NETDEPTH in "${NETDEPTHS[@]}"; do

for NGPUS in "${NUM_GPUS[@]}"; do
    # Create a temporary job script for this particular GPU count
    JOB_SCRIPT="run_script_NETDEPTH${NETDEPTH}_g${NGPUS}.sh"

    cat <<EOF > ${JOB_SCRIPT}
#!/bin/bash
#SBATCH --job-name=gpu_multi_mpi_g${NGPUS}
#SBATCH --partition=gpu_p2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NGPUS}
#SBATCH --gres=gpu:${NGPUS}
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=MC_parallel_${NETDEPTH}_GPUs${NGPUS}.out
#SBATCH --error=MCC_parallel_${NETDEPTH}_GPUs${NGPUS}.err

# Load modules
module purge
module load pytorch-gpu/py3/2.2.0
export PYTHONUSERBASE=/lustre/fswork/projects/rech/emb/ump43xm/torchbraid/torchbraid

# Echo the commands
set -x

srun python -u main_noDP_NI_callibration.py \
					  --batch-size 4 \
					  --epochs 1 \
					  --optimizer SGD \
					  --lr 1e-2 \
					  --momentum .9 \
					  --gradient_accumulation 1 \
					  --model_dimension 128 \
					  --num_heads 1 \
					  --steps ${NETDEPTH} \
					  --Tf ${NETDEPTH} \
					  --lp-cfactor 8 \
					  --lp-max-levels 2 \
					  --lp-fwd-max-iters 2 \
					  --lp-bwd-max-iters 1 \
					  --seed 0 \
					  --scheduler None \
					  --ni_cfactor 2 \
					  --ni_num_levels 1 \
					  --ni_interpolation linear \
					  --ni_interpolate_momentum True
EOF

    # Submit the generated script to Slurm
    sbatch ${JOB_SCRIPT}
done
done
