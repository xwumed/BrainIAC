#!/bin/bash
# BrainIAC feature extraction for UPENN-GBM dataset
# GPU + light CPU (SimpleITK registration + HD-BET), does not conflict with DUNE

# Activate conda
source /home/xuewei/miniforge3/etc/profile.d/conda.sh
conda activate brainiac

# Change to BrainIAC directory (scripts use relative paths for checkpoint)
cd /home/xuewei/MRI/BrainIAC

echo "$(date): Starting BrainIAC UPENN-GBM extraction (671 patients × 4 sequences)"

python extract_features_pipeline.py \
    --input_dir /home/xuewei/MRI/UPENN_flat \
    --output_dir /home/xuewei/MRI/UPENN_flat_brainiac 2>&1

echo "$(date): BrainIAC UPENN-GBM finished. Features at /home/xuewei/MRI/UPENN_flat_brainiac/features.csv"
