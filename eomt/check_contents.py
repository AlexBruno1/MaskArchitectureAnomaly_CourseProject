# eomt/solve_step5.py
import os
import glob
import torch
import numpy as np
import inspect
import sys
from PIL import Image
from argparse import ArgumentParser
import torch.nn.functional as F
from sklearn.metrics import roc_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor

# --- 1. DIRECT IMPORT ---
# We know this exists because of your diagnostic test
try:
    from models.eomt import EoMT
    print("[Check] Imported 'EoMT' class successfully.")
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import EoMT. {e}")
    sys.exit(1)

# --- CONFIGURATION ---
NUM_CLASSES = 19
input_transform = Compose([Resize((512, 1024), Image.BILINEAR), ToTensor()])
target_transform = Compose([Resize((512, 1024), Image.NEAREST)])

# --- 2. METRICS ---
def get_anomaly_scores(pixel_probs):
    scores = {}
    # MSP
    conf, _ = torch.max(pixel_probs, dim=1)
    scores["MSP"] = (1.0 - conf).squeeze(0).cpu().numpy()
    # MaxLogit
    scores["MaxLogit"] = -torch.max(torch.log(pixel_probs + 1e-8), dim=1)[0].squeeze(0).cpu().numpy()
    # MaxEntropy
    scores["MaxEntropy"] = -torch.sum(pixel_probs * torch.log(pixel_probs + 1e-8), dim=1).squeeze(0).cpu().numpy()
    # RbA
    scores["RbA"] = (1.0 - torch.sum(pixel_probs, dim=1)).squeeze(0).cpu().numpy().clip(0, 1)
    return scores

def mask_to_pixel_probs(outputs, h, w):
    # Handle list output (common in some Transformers)
    if isinstance(outputs, (list, tuple)):
        # Usually [logits, masks]
        pred_logits = outputs[0]
        pred_masks = outputs[1]
    else:
        pred_logits = outputs['pred_logits']
        pred_masks = outputs['pred_masks']

    # Softmax classes
    class_probs = F.softmax(pred_logits, dim=-1)
    if class_probs.shape[-1] > NUM_CLASSES: # Remove void/last class
        class_probs = class_probs[..., :-1] 
    
    # Sigmoid masks
    pred_masks = F.interpolate(pred_masks, size=(h, w), mode="bilinear", align_corners=False)
    mask_probs = pred_masks.sigmoid()

    # Combine
    return torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

# --- 3. MAIN ---
def main():
    parser = ArgumentParser()
    parser.add_argument("--dataroot", default="../data/RoadAnomaly_jpg/frames")
    parser.add_argument('--weights', default="eomt_pretrained.pth") 
    args = parser.parse_args()

    device = torch.device('cpu')
    print("--- SOLVING STEP 5: EoMT + RbA ---")

    # A. INITIALIZE MODEL
    print("Initializing EoMT...")
    try:
        # ATTEMPT 1: Standard arguments
        model = EoMT(num_classes=NUM_CLASSES)
        
    except TypeError as e:
        print("\n" + "!"*50)
        print("MODEL INITIALIZATION FAILED (This is expected, don't panic!)")
        print(f"Error: {e}")
        print("-" * 50)
        print("Required Arguments for EoMT:")
        # This prints exactly what the model needs
        print(inspect.signature(EoMT.__init__))
        print("!"*50 + "\n")
        print("ACTION REQUIRED: Copy the line above (starting with 'Required Arguments') and paste it in the chat.")
        return

    model.to(device)
    model.eval()

    # B. LOAD WEIGHTS
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        if 'model' in checkpoint: state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        
        # Clean keys
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.", "").replace("net.", "").replace("module.", "")
            new_state_dict[new_key] = v
            
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"WARNING: Weights '{args.weights}' not found. Using random weights.")

    # C. RUN INFERENCE
    print(f"Scanning {args.dataroot}...")
    valid_pairs = []
    jpg_files = glob.glob(os.path.join(args.dataroot, "**", "*.jpg"), recursive=True)

    for jpg_path in jpg_files:
        label_dir = os.path.basename(jpg_path).replace(".jpg", ".labels")
        parent = os.path.dirname(jpg_path)
        label_path = os.path.join(parent, label_dir, "labels_semantic.png")
        if os.path.exists(label_path):
            valid_pairs.append((jpg_path, label_path))

    results = {"MSP": [], "MaxLogit": [], "MaxEntropy": [], "RbA": []}
    ground_truths = []

    # Process images
    for i, (img_path, label_path) in enumerate(valid_pairs[:10]):
        print(f"[{i+1}] Processing {os.path.basename(img_path)}")
        img = Image.open(img_path).convert('RGB')
        img_tensor = input_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            pixel_probs = mask_to_pixel_probs(outputs, 512, 1024)

        metrics = get_anomaly_scores(pixel_probs)
        for k, v in metrics.items():
            results[k].append(v.flatten())

        gt_mask = np.array(target_transform(Image.open(label_path)))
        ground_truths.append(np.where(gt_mask == 2, 1, 0).flatten())

    # D. PRINT TABLE
    print("\n" + "="*50)
    print(f"{'METHOD':<15} | {'AUPRC':<15} | {'FPR95':<15}")
    print("-" * 50)
    gts = np.concatenate(ground_truths)
    for method in results:
        preds = np.concatenate(results[method])
        auprc = average_precision_score(gts, preds)
        fpr, tpr, _ = roc_curve(gts, preds)
        fpr95 = fpr[np.argmax(tpr >= 0.95)] if np.any(tpr >= 0.95) else 0.0
        print(f"{method:<15} | {auprc:.4f}          | {fpr95:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()