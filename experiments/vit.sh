ds=cifar100
lr=2e-3

### Dense ####
depth=3
width=64
# struct=dense
struct=none
for scale_factor in 1 2 4 6 8 16 32; do
        # --partition=a100_1 \
        # --gres=gpu:a100:1 \
sbatch  --job-name=baseline \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=16 \
        --gres=gpu:1 \
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
--run_name=baseline/${scale_factor}x \
--dataset=${ds} \
--data_path=/scratch/nia4240/struct_sequence_mixing/data/beton \
--checkpoint_folder=/scratch/nia4240/compute-better_spent-scratch/checkpoints \
--model=ViT \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=256 \
--epochs=200 \
--resolution=32 \
--patch_size=8 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--struct=${struct} \
--scheduler=cosine
'"
done;

# ### Kron ####
# depth=3
# width=64
# struct=kron
# layers=all_but_last
# for scale_factor in 1 2 4 8 16 32 64; do
# python3 train_cifar.py \
# --wandb_project=vit_${ds} \
# --dataset=${ds} \
# --model=ViT \
# --width=${width} \
# --depth=${depth} \
# --lr=${lr} \
# --batch_size=256 \
# --epochs=200 \
# --resolution=32 \
# --patch_size=8 \
# --optimizer=adamw \
# --scale_factor=${scale_factor} \
# --struct=${struct} \
# --layers=${layers} \
# --scheduler=cosine
# done;

# ### Low Rank ####
# depth=3
# width=64
# struct=low_rank
# layers=intermediate
# for scale_factor in 1 2 4 8 16 32; do
# python3 train_cifar.py \
# --wandb_project=vit_${ds} \
# --dataset=${ds} \
# --model=ViT \
# --width=${width} \
# --depth=${depth} \
# --lr=${lr} \
# --batch_size=256 \
# --epochs=200 \
# --resolution=32 \
# --patch_size=8 \
# --optimizer=adamw \
# --scale_factor=${scale_factor} \
# --struct=${struct} \
# --rank_frac=0.1 \
# --layers=${layers} \
# --scheduler=cosine
# done;

# ### BTT ####
# depth=3
# width=64
# struct=btt
# layers=all_but_last
# for scale_factor in 1 2 4 8 16 32 64; do
# python3 train_cifar.py \
# --wandb_project=vit_${ds} \
# --dataset=${ds} \
# --model=ViT \
# --width=${width} \
# --depth=${depth} \
# --lr=${lr} \
# --batch_size=256 \
# --epochs=200 \
# --resolution=32 \
# --patch_size=8 \
# --optimizer=adamw \
# --scale_factor=${scale_factor} \
# --struct=${struct} \
# --layers=${layers} \
# --scheduler=cosine
# done;


# # ### Monarch ####
# depth=3
# width=64
# struct=monarch
# layers=all_but_last
# for scale_factor in 0.5 1.5 4 8 12 16; do
# python3 train_cifar.py \
# --wandb_project=vit_${ds} \
# --dataset=${ds} \
# --model=ViT \
# --width=${width} \
# --depth=${depth} \
# --lr=${lr} \
# --batch_size=256 \
# --epochs=200 \
# --resolution=32 \
# --patch_size=8 \
# --optimizer=adamw \
# --scale_factor=${scale_factor} \
# --struct=${struct} \
# --layers=${layers} \
# --scheduler=cosine
# done;

# ### TT ####
# depth=3
# width=64
# struct=tt
# layers=all_but_last
# for scale_factor in 0.5 1 2 4 8 16 32; do
# python3 train_cifar.py \
# --wandb_project=vit_${ds} \
# --dataset=${ds} \
# --model=ViT \
# --width=${width} \
# --depth=${depth} \
# --lr=${lr} \
# --batch_size=256 \
# --epochs=200 \
# --resolution=32 \
# --patch_size=8 \
# --optimizer=adamw \
# --scale_factor=${scale_factor} \
# --struct=${struct} \
# --tt_rank=8 \
# --layers=${layers} \
# --scheduler=cosine
# done;
