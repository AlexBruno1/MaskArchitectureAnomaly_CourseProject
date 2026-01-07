import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from argparse import ArgumentParser

from sklearn.metrics import roc_curve, average_precision_score
import torch.nn.functional as F

from models.eomt import EoMT
from models.vit import ViT
from training.mask_classification_semantic import MaskClassificationSemantic

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 19
MODEL_IMG_SIZE = (1024, 1024)
PATCH_SIZE = 16
NUM_QUERIES = 100
NUM_BLOCKS = 3
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"

def input_transform(img, target_size=MODEL_IMG_SIZE):
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)  # (width, height) for PIL
    img_array = np.array(img, dtype=np.float32)  # Shape: (H, W, C), range [0, 255]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # Shape: (C, H, W)
    return img_tensor

def fpr_at_95_tpr(preds, labels, pos_label=1):
    """Return the FPR when TPR is at minimum 95%.
        
    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)

    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)

def build_model(ckpt_path, device):
    encoder = ViT(
        img_size=MODEL_IMG_SIZE,
        backbone_name=BACKBONE_NAME,
        patch_size=PATCH_SIZE,
    )
    
    network = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=NUM_QUERIES,
        num_blocks=NUM_BLOCKS,
        masked_attn_enabled=True,
    )
    
    model = MaskClassificationSemantic(
        network=network,
        img_size=MODEL_IMG_SIZE,
        num_classes=NUM_CLASSES,
        attn_mask_annealing_enabled=False,
    )
    
    print(f"Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    state_dict = {k: v for k, v in state_dict.items() if "criterion.empty_weight" not in k}
    
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"[WARNING] Missing keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"[WARNING] Unexpected keys: {incompatible.unexpected_keys}")
    
    model.eval()
    model.to(device)
    
    return model

def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )
    parser.add_argument(
        "--ckpt",
        required=True,
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--method",
        default="msp",
        choices=["msp", "maxlogit", "maxentropy", "rba"],
    )

    args = parser.parse_args()

    # CUDA or CPU
    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
        if not args.cpu and not torch.cuda.is_available():
            print("[WARNING] CUDA not available, falling back to CPU")
    else:
        device = "cuda"

    model = build_model(args.ckpt, device)
    print("[OK] EoMT loaded correctly")

    anomaly_scores = []
    anomaly_labels = []

    image_paths = sorted(glob.glob(args.input))

    if len(image_paths) == 0:
        raise RuntimeError("No input images found")

    print(f"Processing {len(image_paths)} images...")

    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        original_size = img.size  # (width, height) - PIL format
        img_tensor = input_transform(img).unsqueeze(0).to(device)  # Shape: (1, C, H, W), range [0, 255]

        with torch.no_grad():
            mask_logits_per_layer, class_logits_per_layer = model(img_tensor)
            
            mask_logits = mask_logits_per_layer[-1]  # Shape: (B, num_queries, H, W)
            class_logits = class_logits_per_layer[-1]  # Shape: (B, num_queries, num_classes+1)
            
            mask_logits = F.interpolate(mask_logits, model.img_size, mode="bilinear")
            
            if args.method == "rba":
                mask_logits_full = F.interpolate(
                    mask_logits,
                    size=(original_size[1], original_size[0]),
                    mode="bilinear",
                    align_corners=False
                )
            
            logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
            
            logits = F.interpolate(
                logits, 
                size=(original_size[1], original_size[0]), 
                mode="bilinear", 
                align_corners=False
            )

        if args.method == "msp":
            probs = torch.softmax(logits, dim=1)
            msp = probs.max(dim=1)[0]
            score = 1.0 - msp.squeeze(0).cpu().numpy()

        elif args.method == "maxlogit":
            score = -np.max(logits.squeeze(0).data.cpu().numpy(), axis=0)

        elif args.method == "maxentropy":
            probs = torch.softmax(logits, dim=1)
            p = probs.squeeze(0).cpu().numpy()
            entropy = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=0)
            score = entropy

        elif args.method == "rba":
            
            mask_probs = mask_logits_full.sigmoid()  # (B, Q, H, W)
            class_probs = class_logits.softmax(dim=-1)[..., :-1]  # (B, Q, num_classes)            
            max_class_probs = class_probs.max(dim=-1)[0]  # (B, Q)
            query_scores = mask_probs * max_class_probs[:, :, None, None]  # (B, Q, H, W)
            max_query_score = query_scores.max(dim=1)[0]  # (B, H, W)
            score = 1.0 - max_query_score

        gt_path = img_path.replace("images", "labels_masks")
        gt_path = os.path.splitext(gt_path)[0] + ".png"

        if not os.path.exists(gt_path):
            print(f"[WARNING] GT not found: {gt_path}")
            continue

        gt = Image.open(gt_path)
        gt = gt.resize(original_size, Image.NEAREST)
        gt = np.array(gt)
        gt = np.where(gt > 0, 1, 0)

        if 1 not in np.unique(gt):
            continue

        anomaly_scores.append(score[gt == 1])
        anomaly_labels.append(np.ones_like(score[gt == 1]))

        anomaly_scores.append(score[gt == 0])
        anomaly_labels.append(np.zeros_like(score[gt == 0]))

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(image_paths)} images")

    if len(anomaly_scores) == 0:
        raise RuntimeError("[ERROR] No valid OOD pixels found")

    scores = np.concatenate(anomaly_scores)
    labels = np.concatenate(anomaly_labels)

    auprc = average_precision_score(labels, scores)
    fpr95 = fpr_at_95_tpr(scores, labels)

    print(f"\n=== RESULTS ({args.method.upper()}) ===")
    print(f"AUPRC: {auprc * 100:.2f}")
    print(f"FPR@95TPR: {fpr95 * 100:.2f}")

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')
    file.write(('    AUPRC score:' + str(auprc*100.0) + '   FPR@TPR95:' + str(fpr95*100.0) ))
    file.write( "\n")
    file.close()

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
