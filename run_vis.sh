set -x

## The official one
python main_vis.py --test --ckpts experiments/pretrain/cfgs/official/ckpt-epoch-300.pth \
                   --config cfgs/pretrain.yaml --exp_name official_epoch_300