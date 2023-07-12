#!/usr/bin/env bash



##GENERAL -----
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --mem=32000M
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

#SBATCH --job-name=train_pid_net
#SBATCH --output=log/%j.out

##DEBUG -----
##SBATCH --partition=debug
##SBATCH --time=00:20:00

##NORMAL -----
#SBATCH --partition=gpu
#SBATCH --time=7-00:00:00
#SBATCH --exclude=gpu[04,02]

module load comp/gcc/11.2.0
module load anaconda
source activate openmmlab

srun python -u tools/train.py $1 --launcher="slurm"
