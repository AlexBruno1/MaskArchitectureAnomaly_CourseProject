# eomt/download_weights.py
import os
import yaml
import glob
from huggingface_hub import hf_hub_download

print("--- EoMT Weight Downloader ---")

# 1. FIND THE CONFIG FILE
# We look for any .yaml file in the configs folder
config_files = glob.glob("configs/*.yaml")

if not config_files:
    print("Error: No config files found in 'eomt/configs/'")
    print("Please make sure you are running this script inside the 'eomt' folder.")
    exit()

# We take the first config found (usually there's only one relevant one for the project)
config_path = config_files[0]
print(f"Reading config: {config_path}")

# 2. READ THE MODEL NAME
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Traverse the YAML structure: trainer -> logger -> init_args -> name
    name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")
    
    if not name:
        print("Error: Could not find 'trainer.logger.init_args.name' in the YAML.")
        print("Please check the file manually.")
        exit()
        
    print(f"Found Model Name: {name}")

except Exception as e:
    print(f"Error parsing YAML: {e}")
    exit()

# 3. DOWNLOAD THE WEIGHTS
repo_id = f"tue-mps/{name}"
filename = "pytorch_model.bin"

print(f"\nAttempting download from: {repo_id}")
print("This might take a few minutes...")

try:
    # Download directly to current folder
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir="." # Save in the current folder
    )
    print("\n" + "="*40)
    print(f"SUCCESS! Weights downloaded to:")
    print(f"{file_path}")
    print("="*40)
    
    # Rename it to make it easy for our other scripts
    new_name = "eomt_pretrained.pth"
    if os.path.exists(filename):
        # Rename pytorch_model.bin -> eomt_pretrained.pth
        try:
            os.rename(filename, new_name)
            print(f"Renamed file to: {new_name}")
            print(f"You can now run 'python solve_step5_final.py --weights {new_name}'")
        except:
            print(f"Could not rename. Use the file '{filename}'")

except Exception as e:
    print(f"\nDOWNLOAD FAILED: {e}")
    print(f"Check if the repo '{repo_id}' actually exists on HuggingFace.")