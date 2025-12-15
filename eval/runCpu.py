# run_cpu.py - Modified for CPU and Inference Only
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor

print("script is running")
# Import the model - Make sure erfnet.py is in the same folder or python path!
try:
    from erfnet import ERFNet
except ImportError:
    # Attempt to fix path if running from root
    import sys
    sys.path.append('./eval') 
    from erfnet import ERFNet

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 20

input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", default="./test_images/*.jpg", nargs="+", help="Input images glob")
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--cpu', action='store_true', help="Force CPU mode")
    args = parser.parse_args()

    # 1. SETUP DEVICE (Fixes the GPU crash)
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Running on CPU.")
    else:
        device = torch.device('cuda')
        print("Running on GPU.")

    # 2. LOAD MODEL
    weightspath = os.path.join(args.loadDir, args.loadWeights)
    print(f"Loading weights from: {weightspath}")

    if not os.path.exists(weightspath):
        print(f"ERROR: Weights file not found at {weightspath}")
        print("Please check the path or download the model.")
        return

    model = ERFNet(NUM_CLASSES)
    
    # Load weights with CPU compatibility
    checkpoint = torch.load(weightspath, map_location=device)
    
    # Handle DataParallel prefix if present in weights
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # 3. RUN INFERENCE
    # Handle input arguments (glob expansion)
    input_files = []
    for pattern in args.input:
        input_files.extend(glob.glob(os.path.expanduser(pattern)))
    
    if not input_files:
        print("No images found! Check your --input path.")
        return

    print(f"Processing {len(input_files)} images...")

    for path in input_files:
        print(f"Processing: {path}")
        
        # Prepare image
        img = Image.open(path).convert('RGB')
        images = input_transform(img).unsqueeze(0).float()
        images = images.to(device) # <--- Fixed: uses dynamic device
        
        # Inference
        with torch.no_grad():
            result = model(images)
        
        # Calculate Anomaly Score
        # (1 - max probability)
        anomaly_result = 1.0 - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)
        
        # Save Result
        output_filename = os.path.basename(path).replace('.', '_result.')
        # Normalize for visualization (0-1 -> 0-255)
        vis_result = (anomaly_result * 255).astype(np.uint8)
        # Apply colormap for better visibility
        vis_color = cv2.applyColorMap(vis_result, cv2.COLORMAP_JET)
        
        cv2.imwrite(f"result_{output_filename}.png", vis_color)
        print(f"Saved result to: result_{output_filename}.png")

        # Skip Ground Truth calculation if files don't exist
        # (Removed the code that crashes if labels are missing)

if __name__ == '__main__':
    main()