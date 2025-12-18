# eomt/solve_step5_all_datasets.py
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

    # --- UPGRADE 1: ReAct ---
    if use_react:
        pred_logits = pred_logits.clip(max=1.0) 
    
    # 2. Softmax (Classes)
    class_probs = F.softmax(pred_logits, dim=-1)
    if class_probs.shape[-1] > NUM_CLASSES: 
        class_probs = class_probs[..., :-1] 
    
    # 3. Sigmoid (Masks)
    pred_masks = F.interpolate(pred_masks, size=(h, w), mode="bilinear", align_corners=False)
    mask_probs = pred_masks.sigmoid()

    # 4. Combine
    return torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

def get_anomaly_scores(pixel_probs):
    scores = {}
    eps = 1e-8
    
    # 1. MSP
    conf, _ = torch.max(pixel_probs, dim=1)
    scores["MSP"] = (1.0 - conf).squeeze(0).cpu().numpy()

    # 2. MaxEntropy
    scores["MaxEntropy"] = -torch.sum(pixel_probs * torch.log(pixel_probs + eps), dim=1).squeeze(0).cpu().numpy()

    # 3. SML Local
    pseudo_logits = torch.log(pixel_probs + eps)
    max_logit, _ = torch.max(pseudo_logits, dim=1)
    mean_l = max_logit.mean()
    std_l = max_logit.std()
    scores["SML_Local"] = -((max_logit - mean_l) / (std_l + eps)).squeeze(0).cpu().numpy()

    # 4. Energy (Used as RbA proxy)
    energy = -torch.logsumexp(pseudo_logits, dim=1)
    scores["Energy"] = energy.squeeze(0).cpu().numpy()
    scores["RbA"] = scores["Energy"] # Mapping for table consistency

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

# --- Helper: Intelligent Path Finder ---
def get_dataset_pairs(name, root_path):
    pairs = []
    print(f"Indexing {name} at {root_path}...")

    # Strategy A: FS Static / FS LostFound (Parallel Folders: 'images' vs 'labels_masks')
    if "fs_static" in name.lower() or "lostfound" in name.lower():
        # Search for images folder
        img_root = os.path.join(root_path, "images")
        if not os.path.exists(img_root):
            # Maybe the user passed the direct 'images' path
            img_root = root_path 
        
        # Grab all PNG and JPG
        images = glob.glob(os.path.join(img_root, "**", "*.png"), recursive=True) + \
                 glob.glob(os.path.join(img_root, "**", "*.jpg"), recursive=True)

        for img_p in images:
            # Replace 'images' with 'labels_masks' in the path
            # This handles subdirectories like /images/subset1/00.png -> /labels_masks/subset1/00.png
            if "images" in img_p:
                lbl_p = img_p.replace("images", "labels_masks")
            else:
                # Fallback if folder isn't named 'images' exactly
                lbl_p = img_p.replace(name, name + "_labels") 

            # FS Labels are usually .png even if image is .jpg
            if img_p.endswith(".jpg"):
                lbl_p = lbl_p.replace(".jpg", ".png")

            if os.path.exists(lbl_p):
                pairs.append((img_p, lbl_p))

    # Strategy B: RoadAnomaly (Nested Folders)
    else:
        images = glob.glob(os.path.join(root_path, "**", "*.jpg"), recursive=True)
        for img_p in images:
            label_dir = os.path.basename(img_p).replace(".jpg", ".labels")
            parent = os.path.dirname(img_p)
            lbl_p = os.path.join(parent, label_dir, "labels_semantic.png")
            if os.path.exists(lbl_p):
                pairs.append((img_p, lbl_p))
                
    print(f"  > Found {len(pairs)} pairs for {name}")
    return pairs

# --- 4. MAIN ---
def main():
    parser = ArgumentParser()
    parser.add_argument('--weights', default="eomt_pretrained.pth") 
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- EVALUATION: RoadAnomaly + FS Static + FS LostFound ---")

    # A. INIT MODEL
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
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        
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

    # --- C. DEFINE DATASETS ---
    # Based on your screenshots, these are the paths:
    dataset_configs = [
        ("RoadAnomaly", "../data/RoadAnomaly_jpg/frames"),
        ("FS_Static", "../data/fs_static"), 
        ("FS_LostFound", "../data/FS_LostFound_full")
    ]

    # --- D. LOOP THROUGH DATASETS ---
    for ds_name, ds_path in dataset_configs:
        print("\n" + "="*80)
        print(f"STARTING EVALUATION: {ds_name}")
        print("="*80)

        valid_pairs = get_dataset_pairs(ds_name, ds_path)
        
        if len(valid_pairs) == 0:
            print(f"Skipping {ds_name} (No images found). Check path: {ds_path}")
            continue

        results = {"MSP": [], "MaxEntropy": [], "SML_Local": [], "Energy": [], "RbA": []}
        ground_truths = []

        for i, (img_path, label_path) in enumerate(valid_pairs): 
            # Progress print every 10 images
            if i % 10 == 0: print(f"[{i+1}/{len(valid_pairs)}] Processing...", end="\r")
            
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = input_transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    pixel_probs = mask_to_pixel_probs(outputs, 512, 1024, use_react=True)

                metrics = get_anomaly_scores(pixel_probs)
                for k, v in metrics.items():
                    results[k].append(v.flatten())

                # Load Ground Truth
                gt_img = Image.open(label_path)
                gt_mask = np.array(target_transform(gt_img))
                
                # FS Datasets usually use 1 for anomaly, RoadAnomaly uses 2. 
                # Let's standardize: Any label > 0 and < 255 is usually anomaly in these benchmarks.
                # But specifically:
                if "RoadAnomaly" in ds_name:
                    ground_truths.append(np.where(gt_mask == 2, 1, 0).flatten())
                else:
                    # FS Static / L&F usually: 0=Road, 1=Anomaly
                    # We treat anything > 0 as anomaly (safest bet for binary eval)
                    ground_truths.append(np.where(gt_mask > 0, 1, 0).flatten())

            except Exception as e:
                print(f"\nError processing {os.path.basename(img_path)}: {e}")
                continue

        # --- TABLE FOR THIS DATASET ---
        print("\n" + "-"*65)
        print(f"RESULTS FOR {ds_name}")
        print(f"{'METHOD':<15} | {'AUPRC':<25} | {'FPR95':<25}")
        print("-" * 65)
        
        if len(ground_truths) > 0:
            gts = np.concatenate(ground_truths)
            for method in results:
                preds = np.concatenate(results[method])
                if np.isnan(preds).any(): preds = np.nan_to_num(preds)

                auprc = average_precision_score(gts, preds)
                fpr, tpr, _ = roc_curve(gts, preds)
                fpr95 = fpr[np.argmax(tpr >= 0.95)] if np.any(tpr >= 0.95) else 0.0
                print(f"{method:<15} | {auprc:.4f}                    | {fpr95:.4f}")
        else:
            print("No valid data processed.")
        print("="*80 + "\n")

if __name__ == '__main__':
    main()