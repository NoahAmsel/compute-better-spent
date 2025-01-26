ds=cifar100
depth=3
width=64
lr=1e-3
scale_factor=64 # 8 # 32
attn="BBTT,2,1,False"  # "none" "\"QK=LowRankQK(heads,dim,dim//heads,attn_mult=attn_mult);VO=LowRankVO(heads,dim,dim//heads,use_bias=use_bias)\""
        # --partition=a100_1 \
        # --gres=gpu:a100:1 \
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
--no-save \
--no-wandb
