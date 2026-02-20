LOGS_DIR=__exps__/__logs__
EXP_ROOT_DIR=__exps__/test
EXP_NAME=test_test

# CONFIG_FP="configs/test.json"
CONFIG_FP="configs/test_categorical.json"

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python test.py \
    --config $CONFIG_FP # > "$LOGS_DIR/_$EXP_NAME.out" 2>&1 &

# ---------------------------------------
# exit