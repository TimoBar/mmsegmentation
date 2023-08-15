#!/usr/bin/env bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:1080:1
#SBATCH --out=log/%j.out
#SBATCH --time=20:00
#SBATCH --mem=10G

module load comp/gcc/11.2.0
module load anaconda
source activate openmmlab
srun python -u tools/analysis_tools/get_flops.py "${@:1}"
