set -x

## The official one
# python main_vis.py --test --ckpts experiments/pretrain/cfgs/official/ckpt-epoch-300.pth \
#                    --config cfgs/pretrain.yaml --exp_name official_epoch_300


# python main_vis.py --test --ckpts experiments/pretrain_VQAE/cfgs/VQAE_1024/ckpt-last.pth \
#                    --config cfgs/pretrain_VQAE.yaml --exp_name VQVE_epoch_215


python main_vis.py --test --ckpts experiments/pretrain_AE_PU1K_h5/cfgs/PU1K_h5_AE_1024/ckpt-last.pth \
                   --config cfgs/pretrain_AE_PU1K_h5.yaml --exp_name VQVE_epoch_215