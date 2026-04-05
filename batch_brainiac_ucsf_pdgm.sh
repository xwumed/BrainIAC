#!/bin/bash
# BrainIAC feature extraction for UCSF-PDGM dataset
# GPU + light CPU (SimpleITK registration + HD-BET), does not conflict with DUNE

# Activate conda
source /home/xuewei/miniforge3/etc/profile.d/conda.sh
conda activate brainiac

# Change to BrainIAC directory (scripts use relative paths for checkpoint)
cd /home/xuewei/MRI/BrainIAC

echo "$(date): Starting BrainIAC UCSF-PDGM extraction (501 patients × 4 sequences)"

python extract_features_pipeline.py \
    --input_dir /home/xuewei/MRI/UCSF_PDGM_flat \
    --output_dir /home/xuewei/MRI/UCSF_PDGM_flat_brainiac 2>&1

echo "$(date): BrainIAC UCSF-PDGM finished. Features at /home/xuewei/MRI/UCSF_PDGM_flat_brainiac/features.csv"
