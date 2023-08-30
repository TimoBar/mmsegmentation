#!/usr/bin/env bash



##GENERAL -----
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --mem=32000M
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

#SBATCH --job-name=test
#SBATCH --output=log/%j_Debug.out

##DEBUG -----
#SBATCH --partition=debug
#SBATCH --time=00:20:00

##NORMAL -----
##SBATCH --partition=gpu
##SBATCH --time=7-00:00:00
##SBATCH --exclude=gpu[04,02]

module load comp/gcc/11.2.0
module load anaconda
source activate openmmlab

port=$(comm -23 <(seq 20000 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

srun python -u tools/test_pruned.py $1 $2 --launcher="slurm" --cfg-options env_cfg.dist_cfg.port=${port}
