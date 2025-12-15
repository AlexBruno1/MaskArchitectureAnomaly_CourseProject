# eomt/solve_step5_final.py
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

# --- 2. INTELLIGENT UNPACKER (THE FIX) ---
def unpack_outputs(outputs):
    """
    Intelligently figures out which output is Masks and which is Logits
    based on dimensionality.
    """
    # 1. If it's a tuple/list, we need to find the tensors
    items = outputs
    
    # If the items are lists (multi-scale), take the last one (final prediction)
    final_items = []
    for item in items:
        if isinstance(item, (list, tuple)):
            final_items.append(item[-1]) # Take last scale
        else:
            final_items.append(item)
            
    # 2. Assign based on Dimensions
    # Masks are [B, Q, H, W] -> 4 Dimensions
    # Logits are [B, Q, C]   -> 3 Dimensions
    
    pred_masks = None
    pred_logits = None
    
    for tensor in final_items:
        if tensor.dim() == 4:
            pred_masks = tensor
        elif tensor.dim() == 3:
            pred_logits = tensor
            
    return pred_logits, pred_masks

def mask_to_pixel_probs(outputs, h, w):
    # 1. Unpack safely
    pred_logits, pred_masks = unpack_outputs(outputs)

    if pred_logits is None or pred_masks is None:
        # Fallback for weird edge cases
        print("Warning: Could not auto-detect shapes. using index 0 as masks.")
        pred_masks = outputs[0][-1]
        pred_logits = outputs[1][-1]

    # 2. Softmax (Classes)
    # pred_logits is [B, Q, 20]. We want [B, Q, 19]
    class_probs = F.softmax(pred_logits, dim=-1)
    if class_probs.shape[-1] > NUM_CLASSES: 
        class_probs = class_probs[..., :-1] 
    
    # 3. Sigmoid (Masks)
    # Resize masks to image size
    pred_masks = F.interpolate(pred_masks, size=(h, w), mode="bilinear", align_corners=False)
    mask_probs = pred_masks.sigmoid()

    # 4. Combine: Sum(ClassProb * MaskProb)
    return torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

def get_anomaly_scores(pixel_probs):
    scores = {}
    conf, _ = torch.max(pixel_probs, dim=1)
    scores["MSP"] = (1.0 - conf).squeeze(0).cpu().numpy()
    scores["MaxLogit"] = -torch.max(torch.log(pixel_probs + 1e-8), dim=1)[0].squeeze(0).cpu().numpy()
    scores["MaxEntropy"] = -torch.sum(pixel_probs * torch.log(pixel_probs + 1e-8), dim=1).squeeze(0).cpu().numpy()
    scores["RbA"] = (1.0 - torch.sum(pixel_probs, dim=1)).squeeze(0).cpu().numpy().clip(0, 1)
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

    device = torch.device('cpu')
    print("--- SOLVING STEP 5: EoMT + RbA (Final) ---")

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

    results = {"MSP": [], "MaxLogit": [], "MaxEntropy": [], "RbA": []}
    ground_truths = []

    # Limit to 5 images for speed, remove [:5] for full run
    for i, (img_path, label_path) in enumerate(valid_pairs[:5]): 
        print(f"[{i+1}/{len(valid_pairs)}] {os.path.basename(img_path)}")
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

    # D. TABLE
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