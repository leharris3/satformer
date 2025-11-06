LOGS_DIR=__exps__/__logs__
EXP_ROOT_DIR=__exps__/test
EXP_NAME="timesformer-cat-loss-cls-weightedCCE-sum-[0-1]-[NOT-LOG]-lr=1e-5-BS=128"

CONFIG_FP="/playpen-ssd/levi/w4c/w4c-25/configs/timesformer/categorical/train_categorical.json"

python train.py \
    --config $CONFIG_FP > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------
exit