#method_names=(knet mask2former maskformer mobilenet_v2 sem_fpn segformer segnext)
# method_names=(mask2former maskformer)
# method_name=${method_names[${SLURM_ARRAY_TASK_ID}]}
dataset=ADE20K
resolution=(512 512)
module load comp/gcc/11.2.0
module load anaconda
source activate openmmlab
python scripts/reproduce_method.py \
    --method_names $@ \
    --dataset ${dataset} \
    --resolution ${resolution[0]} ${resolution[1]} \
    --a100 \
    --submit_slurm_jobs
    #    --method_name ${method_name} \
