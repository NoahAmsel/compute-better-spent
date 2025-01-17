ds=cifar100
depth=3
width=64
for scale_factor in 2 8 32; do
for lr in 5.3e-4 1e-3 1.7e-3 3e-3 5.3e-3; do
for attn in "BBTT,2,1,False"; do  # "none" "\"QK=LowRankQK(heads,dim,dim//heads,attn_mult=attn_mult);VO=LowRankVO(heads,dim,dim//heads,use_bias=use_bias)\""
        # --partition=a100_1 \
        # --gres=gpu:a100:1 \
sbatch  --job-name=BTT_muP_test \
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
--run_name=BTT_muP_test/${scale_factor}x/${lr}/${attn} \
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
--alt_attn_config=${attn} \
--scheduler=cosine \
'"
done;
done;
done;
