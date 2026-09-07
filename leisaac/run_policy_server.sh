#!/bin/bash
PYTHON_BIN="/home/zozo/anaconda3/envs/lerobot/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi
export PYTHONPATH=/home/zozo/lerobot/src:$PYTHONPATH
CUDNN_PATH=$($PYTHON_BIN -c "import os, nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')" 2>/dev/null)
if [ -n "$CUDNN_PATH" ]; then
    export LD_LIBRARY_PATH=$CUDNN_PATH:$LD_LIBRARY_PATH
fi

# Run policy server on host:port
$PYTHON_BIN -m lerobot.async_inference.policy_server --host localhost --port 5555 "$@"


