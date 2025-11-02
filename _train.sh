LOGS_DIR=__exps__/__logs__
EXP_ROOT_DIR=__exps__/test
EXP_NAME=timesformer-reg-bs-1-loss-weightedl1-alpha-3.333

CONFIG_FP="configs/timesformer-bl/train_regression_weighted_l1.json"

python train.py \
    --config $CONFIG_FP > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------
exit