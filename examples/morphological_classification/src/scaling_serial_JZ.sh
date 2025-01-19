#!/bin/bash

# Define arrays of parameters
gpus=(1)
steps=(32 64 128)

# Loop through all combinations of gpus and steps
for n in "${gpus[@]}"; do
  for s in "${steps[@]}"; do
    echo "Submitting job with GPUS=$n and STEPS=$s"

    # Create a temporary SLURM script
    temp_script="slurm_temp_${n}_${s}.sh"
    cat <<EOT > "$temp_script"
#!/bin/bash
#SBATCH --job-name=gpu_multi_mpi     # job name
#SBATCH --partition=gpu_p2           # specify partition
#SBATCH --nodes=1                    # number of nodes (default: 1)
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node (= number of GPUs per node)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=2            # number of CPUs per task (default: 1/4 of the 4-GPU V100 node here)
#SBATCH --hint=nomultithread         # hyperthreading disabled
#SBATCH --time=00:05:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=MC_serial_${n}_${s}_nodes_%j.out  # output file name
#SBATCH --error=MC_serial_${n}_${s}_nodes_%j.out   # error file name (here common with output)

# Cleaning modules loaded in interactive mode and inherited by default
module purge

# Loading modules
module load pytorch-gpu/py3/2.2.0
export PYTHONUSERBASE=/lustre/fswork/projects/rech/emb/ump43xm/torchbraid/torchbraid

# Echo of commands issued
set -x

# Run the script with the specified parameters
srun python -u main_noDP_NI_callibration.py \
  --batch-size 4 \
  --epochs 1 \
  --optimizer SGD \
  --lr 1e-2 \
  --momentum .9 \
  --gradient_accumulation 1 \
  --model_dimension 128 \
  --num_heads 1 \
  --steps $s \
  --Tf $s \
  --lp-cfactor 8 \
  --lp-max-levels 2 \
  --lp-fwd-max-iters 2 \
  --lp-bwd-max-iters 1 \
  --seed 0 \
  --scheduler None \
  --ni_cfactor 2 \
  --ni_num_levels 1 \
  --ni_interpolation linear \
  --ni_interpolate_momentum True \
  --enforce_serial
EOT

    # Submit the job
    sbatch "$temp_script"

    # Optionally remove the temporary script after submission
    rm "$temp_script"
  done
done

