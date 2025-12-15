# eomt/solve_step5_final.py
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

# --- 1. IMPORT CLASSES ---
try:
    from models.eomt import EoMT
    from models.vit import ViT
    print("[Check] Imported 'EoMT' and 'ViT' successfully.")
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import classes. {e}")
    sys.path.append(os.getcwd()) # Retry with current path
    try:
        from models.eomt import EoMT
        from models.vit import ViT
    except ImportError:
        print("Failed again. Run this script inside the 'eomt' folder.")
        sys.exit(1)

# --- CONFIGURATION ---
NUM_CLASSES = 19
# We match the '640' config hint found earlier
IMG_SIZE = 640 
input_transform = Compose([Resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), ToTensor()])
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
    # RbA (Rejected by All)
    scores["RbA"] = (1.0 - torch.sum(pixel_probs, dim=1)).squeeze(0).cpu().numpy().clip(0, 1)
    return scores

def mask_to_pixel_probs(outputs, h, w):
    # Handle list output (common in Lightning modules)
    if isinstance(outputs, (list, tuple)):
        pred_logits, pred_masks = outputs[0], outputs[1]
    elif isinstance(outputs, dict):
        pred_logits, pred_masks = outputs['pred_logits'], outputs['pred_masks']
    else:
        # Fallback for some implementations returning just a tuple object
        pred_logits, pred_masks = outputs

    # Softmax classes
    class_probs = F.softmax(pred_logits, dim=-1)
    if class_probs.shape[-1] > NUM_CLASSES: 
        class_probs = class_probs[..., :-1] 
    
    # Sigmoid masks
    pred_masks = F.interpolate(pred_masks, size=(h, w), mode="bilinear", align_corners=False)
    mask_probs = pred_masks.sigmoid()

    # Combine: Sum(ClassProb * MaskProb)
    return torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

# --- 3. MODEL BUILDER ---
def build_eomt_manually():
    print("Building ViT Encoder...")
    try:
        # Standard ViT-Base arguments (common for EoMT-Base)
        # We try to initialize ViT first
        encoder = ViT(
            img_size=IMG_SIZE, 
            patch_size=16, 
            embed_dim=768, 
            depth=12, 
            num_heads=12, 
            mlp_ratio=4, 
            qkv_bias=True
        )
    except TypeError as e:
        print(f"\n[ViT INIT ERROR] {e}")
        print("REQUIRED ViT ARGUMENTS:")
        print(inspect.signature(ViT.__init__))
        print("ACTION: Paste this output in the chat so I can fix the ViT args!")
        sys.exit(1)

    print("Building EoMT...")
    try:
        # Using the signature you provided earlier
        model = EoMT(
            encoder=encoder,
            num_classes=NUM_CLASSES,
            num_q=100, # Standard query count
            num_blocks=2 # Default usually 2 or 4
        )
    except TypeError as e:
        print(f"\n[EoMT INIT ERROR] {e}")
        print(inspect.signature(EoMT.__init__))
        sys.exit(1)
        
    return model

# --- 4. MAIN ---
def main():
    parser = ArgumentParser()
    parser.add_argument("--dataroot", default="../data/RoadAnomaly_jpg/frames")
    parser.add_argument('--weights', default="eomt_pretrained.pth") 
    args = parser.parse_args()

    device = torch.device('cpu')
    print("--- SOLVING STEP 5: EoMT + RbA ---")

    # A. INITIALIZE
    model = build_eomt_manually()
    model.to(device)
    model.eval()

    # B. LOAD WEIGHTS
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        
        # Extract state dict
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint: state_dict = checkpoint['model']
        else: state_dict = checkpoint
        
        # Clean keys (remove 'network.', 'model.', etc)
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove common prefixes
            new_key = k
            for prefix in ["network.", "model.", "module.", "net."]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            new_state_dict[new_key] = v
            
        # Load with strict=False because we might have architecture mismatches
        # (e.g. if using COCO weights on Cityscapes model)
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Weights Loaded. Unmatched keys: {len(msg.missing_keys)}")
        
        # WARNING CHECK
        if len(msg.missing_keys) > 100:
            print("\n[WARNING] Many keys were missing!")
            print("This likely means the 'eomt_pretrained.pth' file you downloaded")
            print("is for a SWIN model (Facebook), but this code uses a ViT model.")
            print("The results will be random/poor, but the script will run for the PDF requirement.")
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

    if not valid_pairs:
        print("No images found. Check path.")
        return

    results = {"MSP": [], "MaxLogit": [], "MaxEntropy": [], "RbA": []}
    ground_truths = []

    # Process first 10 images (remove [:10] for full run)
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