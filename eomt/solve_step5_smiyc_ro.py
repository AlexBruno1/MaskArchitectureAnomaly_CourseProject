# eomt/solve_step5_smiyc_ro_v2.py
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
target_transform = Compose([Resize((1024, 2048), Image.NEAREST)])

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

def get_anomaly_scores(pixel_probs):
    scores = {}
    conf, _ = torch.max(pixel_probs, dim=1)
    scores["MSP"] = (1.0 - conf).squeeze(0).cpu().numpy()
    scores["MaxLogit"] = -torch.max(torch.log(pixel_probs + 1e-8), dim=1)[0].squeeze(0).cpu().numpy()
    scores["MaxEntropy"] = -torch.sum(pixel_probs * torch.log(pixel_probs + 1e-8), dim=1).squeeze(0).cpu().numpy()
    scores["RbA"] = (1.0 - torch.sum(pixel_probs, dim=1)).squeeze(0).cpu().numpy().clip(0, 1)
    return scores

def build_model_manually():
    print("  > Building ViT (DINOv2) + EoMT...")
    encoder = ViT(img_size=(IMG_SIZE, IMG_SIZE), patch_size=14, backbone_name='vit_large_patch14_reg4_dinov2')
    model = EoMT(encoder=encoder, num_classes=NUM_CLASSES, num_q=100, num_blocks=2, masked_attn_enabled=True)
    return model

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataroot", default="../data/SMIYC_RO21")
    parser.add_argument('--weights', default="eomt_pretrained.pth") 
    args = parser.parse_args()

    device = torch.device('cpu')
    print("--- SOLVING STEP 5: SMIYC ROAD OBSTACLE (V2 Recursive) ---")

    # A. FIND DATA (Improved Logic)
    print(f"Scanning {args.dataroot}...")
    
    # 1. Search for images recursively (inside all subfolders)
    image_dir = os.path.join(args.dataroot, "images")
    extensions = ["*.webp", "*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
    
    print(f"  > Found {len(img_files)} image files in 'images' folder.")

    valid_pairs = []
    missing_labels = 0
    
    for img_path in img_files:
        # Get filename (e.g., 'ca597034.webp')
        filename = os.path.basename(img_path)
        # Convert to label name (e.g., 'ca597034.png')
        label_name = os.path.splitext(filename)[0] + ".png"
        
        # Look for label in 'labels_masks', also recursive
        # We assume labels follow the same folder structure OR are flat
        # Simplest check: Can we find the label file anywhere in labels_masks?
        
        # Construct path: ../data/SMIYC_RO21/labels_masks/label_name
        # BUT: Sometimes images are in subfolders but labels are flat, or vice versa.
        
        # Strategy: direct check
        label_root = os.path.join(args.dataroot, "labels_masks")
        candidate = os.path.join(label_root, label_name)
        
        if os.path.exists(candidate):
            valid_pairs.append((img_path, candidate))
        else:
            # Try searching recursively for the label if not found directly
            found_label = glob.glob(os.path.join(label_root, "**", label_name), recursive=True)
            if found_label:
                valid_pairs.append((img_path, found_label[0]))
            else:
                missing_labels += 1

    if not valid_pairs:
        print("  > CRITICAL: 0 valid pairs found.")
        print(f"  > Checked {len(img_files)} images, but found 0 matching labels.")
        print("  > Please check if your 'labels_masks' folder is empty or has different filenames.")
        return
        
    print(f"  > Successfully paired {len(valid_pairs)} images with labels.")

    # B. INIT MODEL
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

    # C. RUN INFERENCE
    results = {"MSP": [], "MaxLogit": [], "MaxEntropy": [], "RbA": []}
    ground_truths = []

    # Run on first 5 images for quick results
    for i, (img_path, label_path) in enumerate(valid_pairs[:5]): 
        print(f"[{i+1}] {os.path.basename(img_path)}")
        img = Image.open(img_path).convert('RGB')
        img_tensor = input_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            pixel_probs = mask_to_pixel_probs(outputs, 1024, 2048)

        metrics = get_anomaly_scores(pixel_probs)
        for k, v in metrics.items():
            results[k].append(v.flatten())

        gt_mask = np.array(target_transform(Image.open(label_path)))
        # SMIYC RO GT: 1=Anomaly, 0=Normal
        ground_truths.append(np.where(gt_mask == 1, 1, 0).flatten())

    # D. RESULTS TABLE
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