set -x

python main_test.py --config cfgs/pretrain_AE_PU1K_h5.yaml --exp_name test_AE \
                    --test --ckpts experiments/pretrain_AE_PU1K_h5/cfgs/PU1K_h5_AE_1024/ckpt-last.pth

# python main_test.py --config cfgs/pretrain_VQAE_PU1K_h5.yaml --exp_name test_VQAE \
#                     --test --ckpts experiments/pretrain_VQAE_PU1K_h5/cfgs/PU1K_h5_VQVE_1024/ckpt-epoch-250.pth

