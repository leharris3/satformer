export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

LOGS_DIR=__exps__/__logs__
EXP_ROOT_DIR=__exps__/test
EXP_NAME="SaTformer-cat-loss-CCE-lr=1e-5-BS=128-N=64-ATTN=ST^2-baseline"

CONFIG_FP="/playpen-ssd/levi/w4c/w4c-25/configs/SaTformer/categorical/train_categorical.json"

python train.py \
    --config $CONFIG_FP > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------
exit