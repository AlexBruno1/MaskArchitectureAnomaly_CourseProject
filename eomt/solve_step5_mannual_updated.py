# eomt/solve_step5_high_score.py
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

# --- 1. DIRECT IMPORT ---
try:
    from models.eomt import EoMT
    from models.vit import ViT
except ImportError:
    sys.path.append(os.getcwd())
    from models.eomt import EoMT
    from models.vit import ViT

# --- CONFIGURATION ---
NUM_CLASSES = 19
IMG_SIZE = 512 
input_transform = Compose([Resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), ToTensor()])
target_transform = Compose([Resize((512, 1024), Image.NEAREST)])

# --- 2. INTELLIGENT UNPACKER ---
def unpack_outputs(outputs):
    items = outputs
    final_items = []
    for item in items:
        if isinstance(item, (list, tuple)):
            final_items.append(item[-1]) 
        else:
            final_items.append(item)
            
    pred_masks = None
    pred_logits = None
    
    for tensor in final_items:
        if tensor.dim() == 4:
            pred_masks = tensor
        elif tensor.dim() == 3:
            pred_logits = tensor
            
    return pred_logits, pred_masks

def mask_to_pixel_probs(outputs, h, w, use_react=True):
    # 1. Unpack safely
    pred_logits, pred_masks = unpack_outputs(outputs)

    if pred_logits is None or pred_masks is None:
        pred_masks = outputs[0][-1]
        pred_logits = outputs[1][-1]

    # --- UPGRADE 1: ReAct (Rectified Activation) ---
    # We clip the logits at 1.0. This prevents the model from being 
    # "too sure" about the background, allowing anomalies to surface.
    if use_react:
        pred_logits = pred_logits.clip(max=1.0) 
    # -----------------------------------------------

    # 2. Softmax (Classes)
    class_probs = F.softmax(pred_logits, dim=-1)
    if class_probs.shape[-1] > NUM_CLASSES: 
        class_probs = class_probs[..., :-1] 
    
    # 3. Sigmoid (Masks)
    pred_masks = F.interpolate(pred_masks, size=(h, w), mode="bilinear", align_corners=False)
    mask_probs = pred_masks.sigmoid()

    # 4. Combine: Sum(ClassProb * MaskProb)
    return torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

def get_anomaly_scores(pixel_probs):
    scores = {}
    # Small epsilon to prevent log(0)
    eps = 1e-8
    
    # 1. MSP (Baseline)
    conf, _ = torch.max(pixel_probs, dim=1)
    scores["MSP"] = (1.0 - conf).squeeze(0).cpu().numpy()

    # 2. MaxEntropy
    scores["MaxEntropy"] = -torch.sum(pixel_probs * torch.log(pixel_probs + eps), dim=1).squeeze(0).cpu().numpy()

    # --- UPGRADE 2: Local Standardized MaxLogit ---
    # We approximate Logits using Log(Probs).
    # Then we normalize them locally (subtract mean, divide std) 
    # to find pixels that stand out from THIS image's norm.
    pseudo_logits = torch.log(pixel_probs + eps)
    max_logit, _ = torch.max(pseudo_logits, dim=1)
    
    # Standardize
    mean_l = max_logit.mean()
    std_l = max_logit.std()
    scores["SML_Local"] = -((max_logit - mean_l) / (std_l + eps)).squeeze(0).cpu().numpy()

    # --- UPGRADE 3: Energy Score (Replacing broken RbA) ---
    # LogSumExp is a smoother, more robust version of MaxLogit.
    # It acts as a "Free Energy" score.
    energy = -torch.logsumexp(pseudo_logits, dim=1)
    scores["Energy"] = energy.squeeze(0).cpu().numpy()

    return scores

# --- 3. BUILDER ---
def build_model_manually():
    print("  > Building ViT Encoder (DINOv2)...")
    encoder = ViT(
        img_size=(IMG_SIZE, IMG_SIZE),
        patch_size=14,
        backbone_name='vit_large_patch14_reg4_dinov2'
    )
    print("  > Building EoMT Decoder...")
    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=100,
        num_blocks=2,
        masked_attn_enabled=True
    )
    return model

# --- 4. MAIN ---
def main():
    parser = ArgumentParser()
    parser.add_argument("--dataroot", default="../data/RoadAnomaly_jpg/frames")
    parser.add_argument('--weights', default="eomt_pretrained.pth") 
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- SOLVING STEP 5: ReAct + Energy (Optimized) on {device} ---")

    # A. INIT
    try:
        model = build_model_manually()
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[INIT ERROR] {e}")
        return

    # B. LOAD WEIGHTS
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint: state_dict = checkpoint['model']
        else: state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ["network.", "model.", "module.", "net."]:
                if new_key.startswith(prefix): new_key = new_key[len(prefix):]
            new_state_dict[new_key] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        print("Weights Loaded.")
    else:
        print(f"WARNING: '{args.weights}' not found. Using random weights.")

    # C. RUN
    print(f"Scanning {args.dataroot}...")
    valid_pairs = []
    jpg_files = glob.glob(os.path.join(args.dataroot, "**", "*.jpg"), recursive=True)
    for jpg_path in jpg_files:
        label_dir = os.path.basename(jpg_path).replace(".jpg", ".labels")
        parent = os.path.dirname(jpg_path)
        label_path = os.path.join(parent, label_dir, "labels_semantic.png")
        if os.path.exists(label_path):
            valid_pairs.append((jpg_path, label_path))

    results = {"MSP": [], "MaxEntropy": [], "SML_Local": [], "Energy": []}
    ground_truths = []

    # Process all images
    for i, (img_path, label_path) in enumerate(valid_pairs): 
        print(f"[{i+1}/{len(valid_pairs)}] {os.path.basename(img_path)}")
        img = Image.open(img_path).convert('RGB')
        img_tensor = input_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            # ReAct is enabled inside here now
            pixel_probs = mask_to_pixel_probs(outputs, 512, 1024, use_react=True)

        metrics = get_anomaly_scores(pixel_probs)
        for k, v in metrics.items():
            results[k].append(v.flatten())

        gt_mask = np.array(target_transform(Image.open(label_path)))
        # Map label 2 (anomaly) to 1, everything else to 0
        ground_truths.append(np.where(gt_mask == 2, 1, 0).flatten())

    # D. TABLE
    print("\n" + "="*65)
    print(f"{'METHOD':<15} | {'AUPRC (Higher is Better)':<25} | {'FPR95 (Lower is Better)':<25}")
    print("-" * 65)
    
    if len(ground_truths) > 0:
        gts = np.concatenate(ground_truths)
        for method in results:
            preds = np.concatenate(results[method])
            
            # Sanity check for NaNs
            if np.isnan(preds).any():
                print(f"Warning: {method} contains NaNs. Replacing with 0.")
                preds = np.nan_to_num(preds)

            auprc = average_precision_score(gts, preds)
            fpr, tpr, _ = roc_curve(gts, preds)
            fpr95 = fpr[np.argmax(tpr >= 0.95)] if np.any(tpr >= 0.95) else 0.0
            print(f"{method:<15} | {auprc:.4f}                    | {fpr95:.4f}")
    else:
        print("No ground truth data found.")
    print("="*65)

if __name__ == '__main__':
    main()