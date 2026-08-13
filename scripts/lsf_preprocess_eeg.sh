#!/bin/bash
### THINGS-EEG2 raw -> UBP format, one array task per subject.
### Reads  /work3/s193209/data/eeg_raw/sub-XX/ses-0{1..4}/raw_eeg_{training,test}.npy
###   and  <UBP_IMAGE_RESIZE_DIR>/{train,test}_images  (for the img/label/text lists)
### Writes <UBP_DATA_ROOT>/Preprocessed_data_250Hz_whiten/sub-XX/{train,test}.pt
###
### Submit: bsub < scripts/lsf_preprocess_eeg.sh
### (the resize step is already done; chain with -w "done(ubp_resize)" only if redoing it)
### Single subject only:  bsub -J "ubp_prep_eeg[8]" < scripts/lsf_preprocess_eeg.sh
#BSUB -J ubp_prep_eeg[1-10]
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 06:00
#BSUB -o logs/prep_eeg_%J_%I.out
#BSUB -e logs/prep_eeg_%J_%I.err

set -euo pipefail

REPO=/zhome/73/b/145313/UBP
cd "$REPO"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ubp

export UBP_RAW_EEG_DIR=/work3/s193209/data/eeg_raw
export UBP_IMAGE_RESIZE_DIR=/work3/s193209/data/images/Image_set_Resize
export UBP_DATA_ROOT=/work3/s193209/data/eeg_preprocessed_250hz/ubp_format

SUBJECT=${LSB_JOBINDEX:-1}
echo "host=$(hostname) subject=${SUBJECT} start=$(date -Is)"

# Peak RSS is ~25 GB: the float64 epoched/whitened/merged arrays coexist.
# -n 4 x rusage[mem=16GB] reserves 64 GB, so there is headroom.
python preprocess/process_eeg_whiten.py --subject "${SUBJECT}"

echo "done=$(date -Is)"
