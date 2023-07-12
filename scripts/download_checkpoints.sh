#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:1080:1
#SBATCH --out=log/%A_%a.out
#SBATCH --time=20:00
#SBATCH --array=0-5

method_names=(knet mask2former maskformer mobilenet_v2 sem_fpn segformer)
method_name=${method_names[${SLURM_ARRAY_TASK_ID}]}
dataset=ADE20K
resolution=(512 512)
source activate openmmlab
srun python -u scripts/reproduce_method.py \
    --method_name ${method_name} \
    --dataset ${dataset} \
    --resolution ${resolution[0]} ${resolution[1]} \
    --only_download
