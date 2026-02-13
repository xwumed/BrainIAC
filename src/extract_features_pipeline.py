import os
import argparse
import subprocess
import glob
import sys
import pandas as pd
from pathlib import Path

def run_command(cmd):
    """Run a shell command and check for errors."""
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="BrainIAC Feature Extraction Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input MRI images (.nii.gz)")
    parser.add_argument("--output_dir", type=str, default="brainiac_results", help="Directory to save output results (default: brainiac_results)")
    parser.add_argument("--checkpoint", type=str, default="src/checkpoints/BrainIAC.ckpt", help="Path to BrainIAC checkpoint")
    
    args = parser.parse_args()
    
    # Absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    checkpoint = os.path.abspath(args.checkpoint)
    
    # Script locations (assuming this script is in src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocess_script = os.path.join(script_dir, "preprocessing", "mri_preprocess_3d_simple.py")
    feature_script = os.path.join(script_dir, "get_brainiac_features.py")
    template_img = os.path.join(script_dir, "preprocessing", "atlases", "temp_head.nii.gz")
    
    # Setup directories
    unprocessed_dir = os.path.join(output_dir, "data", "unprocessed")
    processed_dir = os.path.join(output_dir, "data", "processed")
    features_csv = os.path.join(output_dir, "features.csv")
    temp_csv = os.path.join(output_dir, "data", "manifest.csv")
    
    os.makedirs(unprocessed_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print("==================================================")
    print("Step 1: Data Preparation And Organization")
    print("==================================================")
    source_files = glob.glob(os.path.join(input_dir, "**", "*.nii.gz"), recursive=True)
    if not source_files:
        print(f"No .nii.gz files found in {input_dir}")
        sys.exit(1)
        
    print(f"Found {len(source_files)} MRI files.")
    # Build mapping: original filename -> (case, sequence)
    file_metadata = {}  # key: original filename (without .nii.gz), value: {case, sequence}
    for src in source_files:
        filename = os.path.basename(src)
        case = os.path.basename(os.path.dirname(src))  # parent folder = case ID
        name_no_ext = filename.replace(".nii.gz", "")
        # Extract sequence from the last part after the last underscore
        sequence = name_no_ext.rsplit("_", 1)[-1] if "_" in name_no_ext else "unknown"
        file_metadata[name_no_ext] = {"case": case, "sequence": sequence}
        dst = os.path.join(unprocessed_dir, filename)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            
    print("==================================================")
    print("Step 2: MRI Preprocessing (Registration + Brain Extraction)")
    print("==================================================")
    cmd_preprocess = [
        sys.executable, preprocess_script,
        "--temp_img", template_img,
        "--input_dir", unprocessed_dir,
        "--output_dir", processed_dir
    ]
    run_command(cmd_preprocess)
    
    print("==================================================")
    print("Step 3: Generating Manifest CSV")
    print("==================================================")
    processed_files = glob.glob(os.path.join(processed_dir, "*.nii.gz"))
    # Filter out temp files if any
    processed_files = [f for f in processed_files if "temp_registered" not in f]
    
    if not processed_files:
        print("No processed files found. Preprocessing might have failed.")
        sys.exit(1)
        
    data = []
    for f in sorted(processed_files):
        pat_id = os.path.basename(f).replace(".nii.gz", "")
        # Match back to original file metadata (processed files have _0000 suffix)
        original_name = pat_id.replace("_0000", "")
        meta = file_metadata.get(original_name, {"case": "unknown", "sequence": "unknown"})
        data.append({"pat_id": pat_id, "label": 0, "case": meta["case"], "sequence": meta["sequence"]})
        
    manifest_df = pd.DataFrame(data)
    manifest_df.to_csv(temp_csv, index=False)
    print(f"Manifest saved to {temp_csv}")
    
    print("==================================================")
    print("Step 4: Feature Extraction")
    print("==================================================")
    cmd_features = [
        sys.executable, feature_script,
        "--checkpoint", checkpoint,
        "--input_csv", temp_csv,
        "--output_csv", features_csv,
        "--root_dir", processed_dir
    ]
    run_command(cmd_features)
    
    print("==================================================")
    print("Step 5: Merging metadata into features")
    print("==================================================")
    features_df = pd.read_csv(features_csv)
    # Prepend case and sequence columns
    features_df.insert(0, "case", manifest_df["case"].values)
    features_df.insert(1, "sequence", manifest_df["sequence"].values)
    features_df.to_csv(features_csv, index=False)
    print(f"Added 'case' and 'sequence' columns to features CSV.")
    
    print("==================================================")
    print("Pipeline Complete!")
    print(f"Features saved to: {features_csv}")
    print("==================================================")

if __name__ == "__main__":
    main()
