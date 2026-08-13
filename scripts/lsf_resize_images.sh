#!/bin/bash
### Resize THINGS images 500x500 -> 224x224 into <UBP_IMAGE_RESIZE_DIR>.
### Already done for the 16740 THINGS images; re-running is a no-op without --overwrite.
### Must finish before lsf_preprocess_eeg.sh (that script reads the resized tree
### to build the img/label/text lists stored in train.pt / test.pt).
### Submit: bsub < scripts/lsf_resize_images.sh
#BSUB -J ubp_resize
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 02:00
#BSUB -o logs/resize_%J.out
#BSUB -e logs/resize_%J.err

set -euo pipefail

REPO=/zhome/73/b/145313/UBP
cd "$REPO"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ubp

export UBP_IMAGE_SRC_DIR=/work3/s193209/data/images
export UBP_IMAGE_RESIZE_DIR=/work3/s193209/data/images/Image_set_Resize

# Idempotent: already-resized images are skipped unless --overwrite is passed.
python preprocess/process_resize.py --type eeg --size 224
