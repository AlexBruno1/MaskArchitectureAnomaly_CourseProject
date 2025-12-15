# eomt/solve_step5_fixed.py
import os
import glob
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import torch.nn.functional as F
from sklearn.metrics import roc_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor
import sys

# --- 1. ROBUST IMPORT SETUP (FIXED) ---
import sys
# Ensure the current directory is in the Python path
sys.path.append(os.getcwd())

try:
    # Attempt 1: Try importing directly from main.py (Most likely location)
    from main import build_model, get_args_parser
    print("[Check] Successfully imported build_model from main.py")
except ImportError:
    try:
        # Attempt 2: Try importing from models.eomt (If defined in the model file)
        from models.eomt import build_model
        from main import get_args_parser
        print("[Check] Successfully imported build_model from models.eomt")
    except ImportError:
        try:
            # Attempt 3: Try generic models package
            from models import build_model
            from main import get_args_parser
            print("[Check] Successfully imported build_model from models package")
        except ImportError as e:
            print(f"\nCRITICAL ERROR: Could not find 'build_model'.")
            print(f"Python Error: {e}")
            print("Please open 'main.py' and check if 'def build_model' exists there.")
            sys.exit(1)

# --- CONFIGURATION ---
NUM_CLASSES = 19 # Cityscapes standard
input_transform = Compose([Resize((512, 1024), Image.BILINEAR), ToTensor()])
target_transform = Compose([Resize((512, 1024), Image.NEAREST)])

# --- 2. METRIC CALCULATIONS (RbA included) ---
def get_anomaly_scores(pixel_probs):
    """Calculates all 4 metrics required by the PDF"""
    scores = {}
    
    # MSP (1 - Max Confidence)
    conf, _ = torch.max(pixel_probs, dim=1)
    scores["MSP"] = (1.0 - conf).squeeze(0).cpu().numpy()

    # MaxLogit (Approximation via Log of Probs)
    scores["MaxLogit"] = -torch.max(torch.log(pixel_probs + 1e-8), dim=1)[0].squeeze(0).cpu().numpy()

    # MaxEntropy
    scores["MaxEntropy"] = -torch.sum(pixel_probs * torch.log(pixel_probs + 1e-8), dim=1).squeeze(0).cpu().numpy()

    # RbA (Rejected by All)
    # Formula: 1.0 - Sum(All Foreground Probabilities)
    scores["RbA"] = (1.0 - torch.sum(pixel_probs, dim=1)).squeeze(0).cpu().numpy().clip(0, 1)
    
    return scores

def mask_to_pixel_probs(outputs, target_h, target_w):
    """Converts MaskFormer outputs to pixel probabilities"""
    pred_logits = outputs['pred_logits']
    class_probs = F.softmax(pred_logits, dim=-1)[..., :-1] # Remove 'void' class
    
    pred_masks = outputs['pred_masks']
    pred_masks = F.interpolate(pred_masks, size=(target_h, target_w), mode="bilinear", align_corners=False)
    mask_probs = pred_masks.sigmoid()

    # Combine: Sum(ClassProb * MaskProb)
    pixel_probs = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    return pixel_probs

# --- 3. MAIN LOGIC ---
def main():
    parser = ArgumentParser()
    parser.add_argument("--dataroot", default="../data/RoadAnomaly_jpg/frames")
    # This matches the argument you were trying to use
    parser.add_argument('--weights', default="eomt_pretrained.pth", help="Path to the .pth file") 
    args = parser.parse_args()

    device = torch.device('cpu')
    print("--- SOLVING STEP 5: EoMT + RbA ---")

    # A. LOAD MODEL
    print("Building Model...")
    try:
        model_args = get_args_parser().parse_args([])
        model_args.device = str(device)
        model_args.num_classes = NUM_CLASSES
        model_args.resume = args.weights
        model_args.eval = True
        
        model, _, postprocessors = build_model(model_args)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error building model: {e}")
        print("Check if 'eomt_pretrained.pth' exists and is a valid checkpoint.")
        return

    # Load Weights Check
    if os.path.exists(args.weights):
        print(f"Loading checkpoint: {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"\n[WARNING] Weight file '{args.weights}' NOT FOUND.")
        print("Please ensure you downloaded the weights to the 'eomt' folder.")
        return

    # B. FIND DATA
    print(f"Scanning {args.dataroot}...")
    valid_pairs = []
    jpg_files = glob.glob(os.path.join(args.dataroot, "**", "*.jpg"), recursive=True)

    for jpg_path in jpg_files:
        # Match RoadAnomaly structure
        label_dir = os.path.basename(jpg_path).replace(".jpg", ".labels")
        parent = os.path.dirname(jpg_path)
        label_path = os.path.join(parent, label_dir, "labels_semantic.png")
        if os.path.exists(label_path):
            valid_pairs.append((jpg_path, label_path))

    print(f"Found {len(valid_pairs)} valid pairs.")
    if not valid_pairs: return

    # C. PROCESSING LOOP
    results = {"MSP": [], "MaxLogit": [], "MaxEntropy": [], "RbA": []}
    ground_truths = []

    # Limit to 5 images for testing
    for i, (img_path, label_path) in enumerate(valid_pairs[:5]):
        print(f"[{i+1}] Processing {os.path.basename(img_path)}")
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = input_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            # Convert to Pixel Map
            pixel_probs = mask_to_pixel_probs(outputs, 512, 1024)

        # Calculate Metrics
        metrics = get_anomaly_scores(pixel_probs)
        for k, v in metrics.items():
            results[k].append(v.flatten())

        # Ground Truth
        gt_mask = np.array(target_transform(Image.open(label_path)))
        ground_truths.append(np.where(gt_mask == 2, 1, 0).flatten())

    # D. PRINT FINAL TABLE
    print("\n" + "="*50)
    print(f"{'METHOD':<15} | {'AUPRC':<15} | {'FPR95':<15}")
    print("-" * 50)
    for method in results:
        gts = np.concatenate(ground_truths)
        preds = np.concatenate(results[method])
        
        auprc = average_precision_score(gts, preds)
        fpr, tpr, _ = roc_curve(gts, preds)
        fpr95 = fpr[np.argmax(tpr >= 0.95)] if np.any(tpr >= 0.95) else 0.0
        
        print(f"{method:<15} | {auprc:.4f}          | {fpr95:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()