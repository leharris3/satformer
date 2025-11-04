LOGS_DIR=__exps__/__logs__
EXP_ROOT_DIR=__exps__/test
EXP_NAME=timesformer-reg-kde-weighted-l1-bs-128

CONFIG_FP="configs/timesformer/train_categorical.json"

python train.py \
    --config $CONFIG_FP # > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------
# exit