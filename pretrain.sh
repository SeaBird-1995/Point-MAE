set -x

################### The official one ######################
# CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/pretrain.yaml --exp_name official

### Ours
CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/pretrain.yaml --num_workers 32 --exp_name official_new_knn
