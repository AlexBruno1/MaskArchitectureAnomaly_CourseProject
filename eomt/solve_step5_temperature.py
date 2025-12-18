# eomt/solve_step5_temperature.py
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

# --- HELPER: UNPACK ---
def unpack_outputs(outputs):
    items = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    final_items = [x[-1] if isinstance(x, (list, tuple)) else x for x in items]
    
    pred_masks, pred_logits = None, None
    for tensor in final_items:
        if tensor.dim() == 4: pred_masks = tensor
        elif tensor.dim() == 3: pred_logits = tensor
    return pred_logits, pred_masks

# --- METRIC: MSP WITH TEMPERATURE ---
def get_msp_temperature(pred_logits, pred_masks, temperature=1.0, target_h=512, target_w=1024):
    """
    Calculates MSP but divides logits by 'temperature' first.
    Formula: Softmax(logits / T)
    """
    # 1. Apply Temperature Scaling
    scaled_logits = pred_logits / temperature

    # 2. Softmax (Classes)
    class_probs = F.softmax(scaled_logits, dim=-1)
    if class_probs.shape[-1] > NUM_CLASSES: 
        class_probs = class_probs[..., :-1] 
    
    # 3. Sigmoid (Masks) - Masks are usually NOT temperature scaled in this context, just class logits
    pred_masks = F.interpolate(pred_masks, size=(target_h, target_w), mode="bilinear", align_corners=False)
    mask_probs = pred_masks.sigmoid()

    # 4. Pixel Probabilities
    pixel_probs = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    
    # 5. Calculate MSP (1 - Max Confidence)
    conf, _ = torch.max(pixel_probs, dim=1)
    msp_score = (1.0 - conf).squeeze(0).cpu().numpy()
    
    return msp_score

# --- BUILDER ---
def build_model_manually():
    print("  > Building ViT Encoder (DINOv2)...")
    encoder = ViT(
        img_size=(IMG_SIZE, IMG_SIZE),
        patch_size=14,
        backbone_name='vit_large_patch14_reg4_dinov2'
    )
    print("  > Building EoMT Decoder...")
    model = EoMT(
        encoder=encoder, num_classes=NUM_CLASSES, num_q=100, num_blocks=2, masked_attn_enabled=True
    )
    return model

# --- MAIN ---
def main():
    parser = ArgumentParser()
    parser.add_argument("--dataroot", default="../data/RoadAnomaly_jpg/frames")
    parser.add_argument('--weights', default="eomt_pretrained.pth") 
    args = parser.parse_args()

    device = torch.device('cpu')
    print("--- SOLVING STEP 5: TEMPERATURE SCALING ---")

    # 1. SETUP
    try:
        model = build_model_manually()
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[INIT ERROR] {e}")
        return

    # 2. LOAD WEIGHTS
    if os.path.exists(args.weights):
        print(f"Loading weights...")
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ["network.", "model.", "module.", "net."]:
                if new_key.startswith(prefix): new_key = new_key[len(prefix):]
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print("Using random weights (Weights not found).")

    # 3. PREPARE DATA
    print(f"Scanning {args.dataroot}...")
    valid_pairs = []
    jpg_files = glob.glob(os.path.join(args.dataroot, "**", "*.jpg"), recursive=True)
    for jpg_path in jpg_files:
        label_dir = os.path.basename(jpg_path).replace(".jpg", ".labels")
        parent = os.path.dirname(jpg_path)
        label_path = os.path.join(parent, label_dir, "labels_semantic.png")
        if os.path.exists(label_path):
            valid_pairs.append((jpg_path, label_path))

    # 4. RUN INFERENCE ONCE, EVALUATE MULTIPLE TEMPS
    results = {
        "MSP (T=1.0)": [],
        "MSP (T=0.5)": [],
        "MSP (T=0.75)": [],
        "MSP (T=1.1)": []
    }
    ground_truths = []

    # run all images
    for i, (img_path, label_path) in enumerate(valid_pairs):
        print(f"[{i+1}] {os.path.basename(img_path)}")
        img = Image.open(img_path).convert('RGB')
        img_tensor = input_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            logits, masks = unpack_outputs(outputs)
            
            # Run for all temperatures without re-running model
            results["MSP (T=1.0)"].append(get_msp_temperature(logits, masks, 1.0).flatten())
            results["MSP (T=0.5)"].append(get_msp_temperature(logits, masks, 0.5).flatten())
            results["MSP (T=0.75)"].append(get_msp_temperature(logits, masks, 0.75).flatten())
            results["MSP (T=1.1)"].append(get_msp_temperature(logits, masks, 1.1).flatten())

        gt_mask = np.array(target_transform(Image.open(label_path)))
        ground_truths.append(np.where(gt_mask == 2, 1, 0).flatten())

    # 5. PRINT TEMPERATURE TABLE
    print("\n" + "="*60)
    print(" TEMPERATURE SCALING RESULTS (Road Anomaly)")
    print("="*60)
    print(f"{'METHOD':<20} | {'AUPRC':<15} | {'FPR95':<15}")
    print("-" * 60)
    
    gts = np.concatenate(ground_truths)
    
    best_score = -1
    best_temp = "None"

    for method in results:
        preds = np.concatenate(results[method])
        auprc = average_precision_score(gts, preds)
        
        # FPR95
        fpr, tpr, _ = roc_curve(gts, preds)
        fpr95 = fpr[np.argmax(tpr >= 0.95)] if np.any(tpr >= 0.95) else 0.0
        
        print(f"{method:<20} | {auprc:.4f}          | {fpr95:.4f}")
        
        if auprc > best_score:
            best_score = auprc
            best_temp = method

    print("-" * 60)
    print(f"Best Temperature: {best_temp}")
    print("="*60)

if __name__ == '__main__':
    main()