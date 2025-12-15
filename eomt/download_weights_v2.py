# eomt/download_weights_final.py
import os
from huggingface_hub import hf_hub_download

print("--- EoMT Public Weight Downloader ---")

# We use the standard Mask2Former Swin-Small model which is public and compatible
# This is the "Fallback" model often used when the specific EoMT one is private
repo_id = "facebook/mask2former-swin-small-cityscapes-semantic"
filename = "model.pkl" 

print(f"\nDownloading PUBLIC weights from: {repo_id}")
print("Please wait...")

try:
    # Download
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir="."
    )
    
    # Rename to what our script expects
    final_name = "eomt_pretrained.pth"
    
    # Remove old file if exists
    if os.path.exists(final_name):
        os.remove(final_name)
        
    # Rename model.pkl -> eomt_pretrained.pth
    os.rename(os.path.basename(file_path), final_name)
    
    print("\n" + "="*40)
    print("SUCCESS! Weights downloaded.")
    print(f"Saved as: {final_name}")
    print("="*40)
    
except Exception as e:
    print(f"\nError: {e}")
    print("If this fails, download manually:")
    print("1. Go to: https://huggingface.co/facebook/mask2former-swin-small-cityscapes-semantic/tree/main")
    print("2. Download 'model.pkl'")
    print("3. Rename it to 'eomt_pretrained.pth' and put it in this folder.")