import argparse
import subprocess
import os
import re
import sys
import glob

def main():
    parser = argparse.ArgumentParser(description="Run Anomaly Evaluation on multiple datasets")
    parser.add_argument("--ckpt_path", required=True, help="Path to the model checkpoint (.ckpt or .bin)")
    parser.add_argument("--dataset_root", required=True, help="Path to the folder containing the datasets (e.g. Validation_Dataset)")
    parser.add_argument("--result_dir", default="result", help="Directory to save the results")
    parser.add_argument("--img_height", type=int, default=640, help="Image height for evaluation")
    parser.add_argument("--img_width", type=int, default=640, help="Image width for evaluation")
    args = parser.parse_args()

    # Create result directory if it doesn't exist
    os.makedirs(args.result_dir, exist_ok=True)

    # Dataset Name -> (Folder Name, Image Extension)
    datasets_config = {
        "SMIYC RA-21": ("RoadAnomaly21", "png"),
        "SMIYC RO-21": ("RoadObsticle21", "webp"),
        "FS L&F": ("FS_LostFound_full", "png"), 
        "FS Static": ("fs_static", "jpg"),
        "Road Anomaly": ("RoadAnomaly", "jpg")
    }
    
    # Order of datasets in the table
    dataset_order = ["SMIYC RA-21", "SMIYC RO-21", "FS L&F", "FS Static", "Road Anomaly"]

    # Methods to evaluate (must match keys in evalAnomalyEomt.py)
    methods_keys = ["MSP", "Max_Logit", "Max_Entropy"]
    
    # Mapping to display names
    method_display_map = {
        "MSP": "MSP",
        "Max_Logit": "MaxLogit",
        "Max_Entropy": "Max Entropy"
    }

    # Store results: results[Method][Dataset][Metric] = value
    results = {
        method: {
            d_name: {"AuPRC": "-", "FPR95": "-"} 
            for d_name in datasets_config.keys()
        } 
        for method in methods_keys
    }

    # Path to the evaluation script
    # Assuming this script is run from the project root
    script_path = os.path.join("eval", "evalAnomalyEomt.py")
    if not os.path.exists(script_path):
        print(f"Error: Could not find {script_path}. Please run this script from the project root.")
        return

    # 1. Iterate over datasets and run evaluation
    for d_name in dataset_order:
        folder, ext = datasets_config[d_name]
        print(f"\nProcessing dataset: {d_name} ({folder})...")
        
        # Construct path to images
        # glob pattern passed as string to the script
        input_pattern = os.path.join(args.dataset_root, folder, "images", f"*.{ext}")
        
        # Verify if any files exist matching the pattern to fail early/warn
        # (The script glob logic is inside evalAnomalyEomt, but good to check here)
        files = glob.glob(input_pattern)
        if not files:
            print(f"Warning: No files found for pattern {input_pattern}")
            # We continue, let the subprocess handle it or fail gracefully
        else:
            print(f"Found {len(files)} files.")

        # Construct command
        cmd = [
            sys.executable, script_path,
            "--ckpt_path", args.ckpt_path,
            "--input", input_pattern,
            "--img_height", str(args.img_height),
            "--img_width", str(args.img_width),
            # We redirect output to a temp file or just capture stdout.
            # We let the script write to its default result file alongside stdout, but we parse stdout here.
            "--result_file", os.path.join(args.result_dir, f"temp_res_{folder}.txt") 
        ]

        print(f"Command: {' '.join(cmd)}")

        try:
            # Run the command and capture output
            process = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            # Print stdout for user to see progress
            print(process.stdout)
            
            if process.returncode != 0:
                print(f"Error running evaluation for {d_name}.")
                print("Stderr:", process.stderr)
                continue

            # Parse results from stdout
            output_lines = process.stdout.splitlines()
            for line in output_lines:
                # Example lines:
                # [MSP] AUPRC score: 85.3
                # [MSP] FPR@95TPR: 12.5
                
                # Regex for AUPRC
                auprc_match = re.search(r"\[(.*?)\] AUPRC score:\s*([\d\.]+)", line)
                if auprc_match:
                    m_key = auprc_match.group(1)
                    val = float(auprc_match.group(2))
                    if m_key in results:
                        results[m_key][d_name]["AuPRC"] = f"{val:.2f}"

                # Regex for FPR
                fpr_match = re.search(r"\[(.*?)\] FPR@95TPR:\s*([\d\.]+)", line)
                if fpr_match:
                    m_key = fpr_match.group(1)
                    val = float(fpr_match.group(2))
                    if m_key in results:
                        results[m_key][d_name]["FPR95"] = f"{val:.2f}"
                        
        except Exception as e:
            print(f"Exception during execution for {d_name}: {e}")

    # 2. Generate Table
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]
    output_file = os.path.join(args.result_dir, f"{ckpt_name}.csv")
    print("\ngenerating results table...")
    
    with open(output_file, "w") as f:
        # Header Row 1
        header1 = "Model,Method,mIoU"
        for d in dataset_order:
            header1 += f",{d},{d}"
        f.write(header1 + "\n")
        
        # Header Row 2
        header2 = ",,," # Skip Model, Method, mIoU columns
        for _ in dataset_order:
            header2 += "AuPRC,FPR95,"
        # Remove trailing comma
        header2 = header2.rstrip(",")
        f.write(header2 + "\n")
        
        # Data Rows
        # Model is always "EoMT" based on context (or inferred from ckpt)
        model_name = "EoMT"
        
        for m_key in methods_keys:
            display_name = method_display_map[m_key]
            row = f"{model_name},{display_name}," # mIoU is empty
            
            for d in dataset_order:
                val_auprc = results[m_key][d]["AuPRC"]
                val_fpr = results[m_key][d]["FPR95"]
                row += f"{val_auprc},{val_fpr},"
            
            row = row.rstrip(",")
            f.write(row + "\n")
            
    # Also print a pretty table to stdout
    print("\n--- Results ---")
    header_fmt = "{:<10} {:<15} {:<8}" + "".join([ "{:<25}" for _ in dataset_order ])
    print(header_fmt.format("Model", "Method", "mIoU", *dataset_order))
    
    sub_header_fmt = "{:<10} {:<15} {:<8}" + "".join([ "{:<12} {:<13}" for _ in dataset_order ])
    sub_header_vals = ["", "", ""]
    for _ in dataset_order:
        sub_header_vals.extend(["AuPRC", "FPR95"])
    print(sub_header_fmt.format(*sub_header_vals))
    
    for m_key in methods_keys:
        display_name = method_display_map[m_key]
        vals = [model_name, display_name, "-"]
        for d in dataset_order:
            vals.append(results[m_key][d]["AuPRC"])
            vals.append(results[m_key][d]["FPR95"])
        print(sub_header_fmt.format(*vals))

    print(f"\nTable saved to {output_file}")

if __name__ == "__main__":
    main()
