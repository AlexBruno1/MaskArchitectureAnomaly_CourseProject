# eomt/solve_step6_extension.py
import os
import glob
import torch
import numpy as np
import sys
from PIL import Image
from argparse import ArgumentParser
import torch.nn.functional as F
from sklearn.metrics import roc_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor

# --- IMPORTS ---
try:
    from models.eomt import EoMT
    from models.vit import ViT
except ImportError:
    sys.path.append(os.getcwd())
    from models.eomt import EoMT
    from models.vit import ViT

# --- CONFIG ---
NUM_CLASSES = 19
IMG_SIZE = 512
input_transform = Compose([Resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), ToTensor()])
target_transform = Compose([Resize((512, 1024), Image.NEAREST)])

# --- HELPER FUNCTIONS ---
def unpack_outputs(outputs):
    items = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    final_items = [x[-1] if isinstance(x, (list, tuple)) else x for x in items]
    pred_masks, pred_logits = None, None
    for tensor in final_items:
        if tensor.dim() == 4: pred_masks = tensor
        elif tensor.dim() == 3: pred_logits = tensor
    return pred_logits, pred_masks

def mask_to_pixel_probs(outputs, h, w):
    pred_logits, pred_masks = unpack_outputs(outputs)
    class_probs = F.softmax(pred_logits, dim=-1)
    if class_probs.shape[-1] > NUM_CLASSES: class_probs = class_probs[..., :-1] 
    pred_masks = F.interpolate(pred_masks, size=(h, w), mode="bilinear", align_corners=False)
    mask_probs = pred_masks.sigmoid()
    return torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

def normalize_score(score_map):
    """Normalize scores to 0-1 range so they can be combined."""
    min_v = score_map.min()
    max_v = score_map.max()
    if max_v - min_v == 0: return score_map
    return (score_map - min_v) / (max_v - min_v)

def get_anomaly_scores_extended(pixel_probs):
    scores = {}
    
    # 1. Standard RbA
    rba = (1.0 - torch.sum(pixel_probs, dim=1)).squeeze(0).cpu().numpy()
    scores["RbA"] = rba
    
    # 2. Standard MSP
    conf, _ = torch.max(pixel_probs, dim=1)
    msp = (1.0 - conf).squeeze(0).cpu().numpy()
    scores["MSP"] = msp
    
    # 3. EXTENSION: Combined Score (Ensemble)
    # Normalize both and average them
    norm_rba = normalize_score(rba)
    norm_msp = normalize_score(msp)
    
    scores["Ensemble (MSP+RbA)"] = (norm_rba + norm_msp) / 2.0
    
    return scores

# --- BUILDER ---
def build_model_manually():
    print("  > Building ViT (DINOv2) + EoMT...")
    encoder = ViT(img_size=(IMG_SIZE, IMG_SIZE), patch_size=14, backbone_name='vit_large_patch14_reg4_dinov2')
    model = EoMT(encoder=encoder, num_classes=NUM_CLASSES, num_q=100, num_blocks=2, masked_attn_enabled=True)
    return model

# --- MAIN ---
def main():
    parser = ArgumentParser()
    parser.add_argument("--dataroot", default="../data/RoadAnomaly_jpg/frames") # Use your working dataset
    parser.add_argument('--weights', default="eomt_pretrained.pth") 
    args = parser.parse_args()

    device = torch.device('cpu')
    print("--- SOLVING STEP 6: PROJECT EXTENSION (Ensemble Analysis) ---")

    # A. INIT
    try:
        model = build_model_manually()
        model.to(device)
        model.eval()
        if os.path.exists(args.weights):
            print("Loading weights...")
            checkpoint = torch.load(args.weights, map_location=device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            new_state_dict = {k.replace("model.", "").replace("network.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print(f"[INIT ERROR] {e}")
        return

    # B. DATA
    print(f"Scanning {args.dataroot}...")
    valid_pairs = []
    jpg_files = glob.glob(os.path.join(args.dataroot, "**", "*.jpg"), recursive=True)
    for jpg_path in jpg_files:
        label_dir = os.path.basename(jpg_path).replace(".jpg", ".labels")
        parent = os.path.dirname(jpg_path)
        label_path = os.path.join(parent, label_dir, "labels_semantic.png")
        if os.path.exists(label_path):
            valid_pairs.append((jpg_path, label_path))

    results = {"RbA": [], "MSP": [], "Ensemble (MSP+RbA)": []}
    ground_truths = []

    # C. RUN
    # Limit to 5 images for speed
    for i, (img_path, label_path) in enumerate(valid_pairs[:10]): 
        print(f"[{i+1}] Processing {os.path.basename(img_path)}...")
        img = Image.open(img_path).convert('RGB')
        img_tensor = input_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            pixel_probs = mask_to_pixel_probs(outputs, 512, 1024)

        metrics = get_anomaly_scores_extended(pixel_probs)
        for k, v in metrics.items():
            results[k].append(v.flatten())

        gt_mask = np.array(target_transform(Image.open(label_path)))
        ground_truths.append(np.where(gt_mask == 2, 1, 0).flatten())

    # D. TABLE
    print("\n" + "="*60)
    print(" EXTENSION RESULTS: METHOD FUSION")
    print("="*60)
    print(f"{'METHOD':<25} | {'AUPRC':<15} | {'FPR95':<15}")
    print("-" * 60)
    gts = np.concatenate(ground_truths)
    for method in results:
        preds = np.concatenate(results[method])
        auprc = average_precision_score(gts, preds)
        fpr, tpr, _ = roc_curve(gts, preds)
        fpr95 = fpr[np.argmax(tpr >= 0.95)] if np.any(tpr >= 0.95) else 0.0
        print(f"{method:<25} | {auprc:.4f}          | {fpr95:.4f}")
    print("="*60)
    print("\nINTERPRETATION:")
    print("If 'Ensemble' is higher than individual scores, the fusion worked.")
    print("This demonstrates a low-cost method to improve performance.")

if __name__ == '__main__':
    main()