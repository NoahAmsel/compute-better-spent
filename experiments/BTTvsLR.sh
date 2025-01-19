ds=cifar100
depth=3
width=64
for scale_factor in 1 2 4 8 16 32; do # 8 32 128 256
for lr in 2e-3; do
for attn in  "none" "\"QK=LowRankQK(heads,dim,dim//heads,attn_mult=attn_mult);VO=LowRankVO(heads,dim,dim//heads,use_bias=use_bias)\"" "BBTT,2,1,False"; do
sbatch  --job-name=BTTvsLR3 \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=16 \
        --gres=gpu:a100:1 \
        --time=02:00:00 \
        --mem=32G \
        --output=/scratch/nia4240/compute-better_spent-scratch/slurm/%x/%j.out \
        --wrap="singularity exec --nv --overlay /scratch/nia4240/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c '
source /ext3/env.sh;
conda activate struct_orig;
cd /home/nia4240/compute-better-spent;
export WANDB__SERVICE_WAIT=300;
python3 train_cifar.py \
--wandb_project=struct_sequence_mixing \
--run_name=BTTvsLR3/${scale_factor}x/${lr}/${attn} \
--dataset=${ds} \
--data_path=/scratch/nia4240/struct_sequence_mixing/data/beton \
--checkpoint_folder=/scratch/nia4240/compute-better_spent-scratch/checkpoints \
--model=ViT \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=256 \
--epochs=1000 \
--resolution=32 \
--patch_size=8 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--alt_attn_config=${attn} \
--scheduler=cosine
'"
done;
done;
done;
