#!/bin/bash
### Train the UBP EEG encoder: EEGProjectLayer + ViT-H-14 (z_dim 1024), one subject per array task.
###
### Outputs -> /work3/s193209/data/ubp_exp/<name>/sub-XX_seed0/
### CLIP feature cache -> /work3/.../ubp_format/Image_feature/FoveaBlur/ (built on first run)
###
### The CLIP feature cache is keyed by config name + split, NOT by subject, so a
### 10-wide array would have every task encode the same ~50k blurred images through
### ViT-H-14. Warm it with one subject first, then fan out:
###   bsub -J "ubp_train[1]" < scripts/lsf_train_eeg.sh
###   bsub -w "done(ubp_train)" -J "ubp_train[2-10]" < scripts/lsf_train_eeg.sh
###
### All 10 at once (wasteful but safe, the cache write is atomic):
###   bsub < scripts/lsf_train_eeg.sh
### Baseline instead of UBP: CONFIG=configs/eeg/baseline.yaml bsub < scripts/lsf_train_eeg.sh
### Inter-subject (leave-one-out, lr defaults to 1e-5):
###   EXP_SETTING=inter-subject bsub < scripts/lsf_train_eeg.sh
#BSUB -J ubp_train[1-10]
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 04:00
#BSUB -o logs/train_%J_%I.out
#BSUB -e logs/train_%J_%I.err

set -euo pipefail

REPO=/zhome/73/b/145313/UBP
cd "$REPO"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ubp

# Large model weights must never land in $HOME (no quota for them there).
export HF_HOME=/work3/s193209/huggingface_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME=/work3/s193209/torch_cache
# ViT-H-14 laion2b_s32b_b79k is already cached, so no download is needed.
export HF_HUB_OFFLINE=1

# LSF hands us one GPU via CUDA_VISIBLE_DEVICES; torch sees it as index 0.
# Without this, get_device('auto') would parse nvidia-smi's physical index.
export UBP_GPU=0

CONFIG=configs/eeg/ubp_allch.yaml
EXP_SETTING=${EXP_SETTING:-intra-subject}
BRAIN_BACKBONE=${BRAIN_BACKBONE:-EEGProjectLayer}
VISION_BACKBONE=${VISION_BACKBONE:-ViT-H-14}
EPOCH=${EPOCH:-50}
SEED=${SEED:-0}
# intra-subject trains on 1e-4; inter-subject (leave-one-out) on 1e-5.
if [ "$EXP_SETTING" = "inter-subject" ]; then
    LR=${LR:-1e-5}
else
    LR=${LR:-1e-4}
fi

SUBJECT=$(printf "%02d" "${LSB_JOBINDEX:-1}")
echo "host=$(hostname) subject=sub-${SUBJECT} config=${CONFIG} setting=${EXP_SETTING} lr=${LR} start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python main.py \
    --config "$CONFIG" \
    --dataset eeg \
    --subjects "sub-${SUBJECT}" \
    --seed "$SEED" \
    --exp_setting "$EXP_SETTING" \
    --brain_backbone "$BRAIN_BACKBONE" \
    --vision_backbone "$VISION_BACKBONE" \
    --epoch "$EPOCH" \
    --lr "$LR"

echo "done=$(date -Is)"
