#!/bin/bash
### Serve TensorBoard for the UBP runs on the current login node.
###
###   /bin/bash scripts/tensorboard.sh [PORT]
###
### Then from your laptop, tunnel to the SAME login node this runs on:
###   ssh -L 6006:localhost:6006 s193209@login.hpc.dtu.dk
### and open http://localhost:6006
###
### Why the PYTHONPATH dance: the `ubp` env has tensorboard 2.14, which is
### incompatible with numpy 2 (`np.string_` was removed) and with setuptools>=81
### (no pkg_resources). Rather than upgrade tensorboard inside `ubp` -- which would
### swap files under any running training job -- a newer tensorboard is installed
### standalone on /work3 and put ahead of the env on sys.path. TB 2.19 is pinned
### because 2.21's protobuf gencode needs protobuf 6.x while the env has 5.28.
### Everything else (numpy, grpcio, werkzeug, tensorboard-data-server) comes from `ubp`.

set -euo pipefail

PORT=${1:-6006}
LOGDIR=${LOGDIR:-/work3/s193209/data/ubp_exp}
TB_PREFIX=/work3/s193209/pyenvs/tb
PYBIN="$HOME/miniforge3/envs/ubp/bin/python"

if [ ! -d "$TB_PREFIX" ]; then
    echo "installing standalone tensorboard to $TB_PREFIX"
    "$PYBIN" -m pip install -q --no-deps --target "$TB_PREFIX" "tensorboard==2.19.0"
fi

echo "logdir : $LOGDIR"
echo "host   : $(hostname)  (tunnel to this exact login node)"
echo "url    : http://localhost:${PORT} after: ssh -L ${PORT}:localhost:${PORT} $USER@login.hpc.dtu.dk"

PYTHONPATH="$TB_PREFIX" exec "$PYBIN" -m tensorboard.main \
    --logdir "$LOGDIR" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --reload_interval 30
