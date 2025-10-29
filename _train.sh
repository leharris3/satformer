export CUDA_VISIBLE_DEVICES=0

LOGS_DIR=__exps__/__logs__
EXP_ROOT_DIR=__exps__/test
EXP_NAME=initial_test

CONFIG_FP="configs/train.json"

python train.py \
    --config $CONFIG_FP  > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------
exit