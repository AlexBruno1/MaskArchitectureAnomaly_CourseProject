# eomt/solve_miou.py
import os
import glob
import torch
import numpy as np
import sys
from PIL import Image
from argparse import ArgumentParser
import torch.nn.functional as F
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

# CITYSCAPES CLASS MAPPING (ID -> TrainID)
ID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
    26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
}

def map_label(label_img):
    label_arr = np.array(label_img, dtype=np.uint8)
    label_copy = 255 * np.ones_like(label_arr, dtype=np.uint8) # 255 is 'ignore'
    for k, v in ID_TO_TRAINID.items():
        label_copy[label_arr == k] = v
    return label_copy

def unpack_outputs(outputs):
    items = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    final_items = [x[-1] if isinstance(x, (list, tuple)) else x for x in items]
    pred_masks, pred_logits = None, None
    for tensor in final_items:
        if tensor.dim() == 4: pred_masks = tensor
        elif tensor.dim() == 3: pred_logits = tensor
    return pred_logits, pred_masks

def mask_to_prediction(outputs, h, w):
    pred_logits, pred_masks = unpack_outputs(outputs)
    
    class_probs = F.softmax(pred_logits, dim=-1)
    if class_probs.shape[-1] > NUM_CLASSES: 
        class_probs = class_probs[..., :-1] 
    
    pred_masks = F.interpolate(pred_masks, size=(h, w), mode="bilinear", align_corners=False)
    mask_probs = pred_masks.sigmoid()
    
    # Combine to get per-pixel class scores
    sem_seg = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    return torch.argmax(sem_seg, dim=1).squeeze(0).cpu().numpy()

def build_model_manually():
    print("  > Building ViT (DINOv2) + EoMT...")
    encoder = ViT(img_size=(IMG_SIZE, IMG_SIZE), patch_size=14, backbone_name='vit_large_patch14_reg4_dinov2')
    model = EoMT(encoder=encoder, num_classes=NUM_CLASSES, num_q=100, num_blocks=2, masked_attn_enabled=True)
    return model

def main():
    parser = ArgumentParser()
    # Updated default path to match your screenshot structure
    parser.add_argument("--dataroot", default="../data") 
    parser.add_argument('--weights', default="eomt_pretrained.pth") 
    args = parser.parse_args()

    device = torch.device('cpu')
    print("--- CALCULATING mIoU ON CITYSCAPES ---")

    # 1. SETUP PATHS (Adjusted for your folders)
    # Your structure: data/leftImg8bit_trainvaltest/leftImg8bit/val
    img_root = os.path.join(args.dataroot, "leftImg8bit_trainvaltest", "leftImg8bit", "val")
    lbl_root = os.path.join(args.dataroot, "gtFine_trainvaltest", "gtFine", "val")
    
    print(f"Looking for images in: {img_root}")
    
    # 2. LOAD MODEL
    try:
        model = build_model_manually()
        model.to(device)
        model.eval()
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace("model.", "").replace("network.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print(f"[INIT ERROR] {e}")
        return

    # 3. SCAN FILES
    img_files = glob.glob(os.path.join(img_root, "**", "*leftImg8bit.png"), recursive=True)
    valid_pairs = []
    
    for img_path in img_files:
        # File: frankfurt_000000_000294_leftImg8bit.png
        # Goal: frankfurt_000000_000294_gtFine_labelIds.png
        basename = os.path.basename(img_path)
        city = os.path.basename(os.path.dirname(img_path))
        
        label_name = basename.replace("leftImg8bit.png", "gtFine_labelIds.png")
        label_path = os.path.join(lbl_root, city, label_name)
        
        if os.path.exists(label_path):
            valid_pairs.append((img_path, label_path))
            
    print(f"Found {len(valid_pairs)} valid image/label pairs.")
    if len(valid_pairs) == 0:
        print("CRITICAL ERROR: No pairs found.")
        print("Check if 'gtFine_trainvaltest' is extracted correctly in ../data/")
        return

    # 4. RUN METRICS
    intersection_meter = np.zeros(NUM_CLASSES)
    union_meter = np.zeros(NUM_CLASSES)
    
    # Process 50 images to get a quick estimate (Remove [:50] for full accuracy)
    # Using full set takes ~30 mins on CPU. 50 images takes ~3 mins.
    print("Running on full validation set (500 images)... This may take 20-40 mins.")
    
    for i, (img_path, label_path) in enumerate(valid_pairs): 
        if i % 10 == 0: print(f"Processing {i}/{len(valid_pairs)}...")
        
        # Load & Prep
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img_tensor = input_transform(img).unsqueeze(0).to(device)
        
        # Infer
        with torch.no_grad():
            outputs = model(img_tensor)
            pred = mask_to_prediction(outputs, h, w)
            
        # Ground Truth
        lbl = Image.open(label_path)
        target = map_label(lbl)
        
        # Accumulate IoU
        for cls in range(NUM_CLASSES):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            intersection = (pred_inds & target_inds).sum()
            union = (pred_inds | target_inds).sum()
            
            intersection_meter[cls] += intersection
            union_meter[cls] += union

    # 5. FINAL CALCULATION
    iou_per_class = intersection_meter / (union_meter + 1e-10)
    miou = np.mean(iou_per_class)
    
    print("\n" + "="*40)
    print(f"FINAL RESULT: mIoU = {miou * 100:.2f}%")
    print("="*40)

if __name__ == '__main__':
    main()