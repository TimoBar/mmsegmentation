#!/usr/bin/env bash



##GENERAL -----
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=32000M
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

#SBATCH --job-name=train
#SBATCH --output=log/%j.out

##DEBUG -----
##SBATCH --partition=debug
##SBATCH --time=00:20:00

##NORMAL -----
#SBATCH --partition=gpu,gpub
#SBATCH --time=7-00:00:00
##SBATCH --exclude=gpu[04,02]

module load comp/gcc/11.2.0
module load anaconda
source activate openmmlab

port=$(comm -23 <(seq 30000 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

srun dlprof --mode pytorch python -u tools/test.py $1 $2 --launcher="slurm" --cfg-options env_cfg.dist_cfg.port=${port} test_dataloader.num_workers=0 val_dataloader.num_workers=0 train_cfg.max_iters=300 "${@:4}"
