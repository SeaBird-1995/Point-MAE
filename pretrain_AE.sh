set -x

# CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pretrain_AE.yaml --num_workers 16 --exp_name AE_1024

# CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/pretrain_VQAE.yaml --num_workers 16 --exp_name VQAE_1024

CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/pretrain_AE_PU1K.yaml --num_workers 16 --exp_name PU1K_AE_1024

# CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/pretrain_VQAE_PU1K.yaml --num_workers 16 --exp_name PU1K_VQAE_1024
