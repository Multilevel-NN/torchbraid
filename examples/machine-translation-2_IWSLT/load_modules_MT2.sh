module load daint-gpu cray-python 
export MPICC=cc
# conda activate /scratch/snx3000/msalvado/envs/tb3_18_02_2024
conda activate original-transf_MT_20240717
export MPICC=cc
export MPICH_RDMA_ENABLED_CUDA=1
