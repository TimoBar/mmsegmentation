#!/usr/bin/env bash



##GENERAL -----
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --mem=32000M
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

#SBATCH --job-name=mmseg_test
#SBATCH --output=mmseg_test.out

##DEBUG -----
#SBATCH --partition=debug
#SBATCH --time=00:20:00

##NORMAL -----
##SBATCH --partition=gpu,gpub
##SBATCH --time=7-00:00:00
##SBATCH --exclude=gpu[04,02]

# Load modules
module load comp/gcc/11.2.0
source activate openmmlab

# Extra output
nvidia-smi
echo -e "Node: $(hostname)"
echo -e "Job internal GPU id(s): $CUDA_VISIBLE_DEVICES"
echo -e "Job external GPU id(s): ${SLURM_JOB_GPUS}"

srun python -u demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --out-file result.jpg

