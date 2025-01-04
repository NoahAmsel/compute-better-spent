ds=cifar100
lr=1e-3
depth=3
width=64
struct=dense
for scale_factor in 1; do  # 0.5 1 2 4 6 8 10 12
python3 train_cifar.py \
--wandb_project=struct_sequence_mixing \
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
done;
