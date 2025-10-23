export CUDA_VISIBLE_DEVICES=5

LOGS_DIR=__exps__/__logs__
EXP_ROOT_DIR=__exps__/bto
EXP_NAME=long-run-mb-stats-psnr-ssim-test

CONFIG_FP="_test.yaml"

python train_simple.py \
    --config $CONFIG_FP > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------
exit