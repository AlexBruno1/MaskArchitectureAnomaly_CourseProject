import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import torch.nn.functional as F
from sklearn.metrics import roc_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms import InterpolationMode

from models.eomt import EoMT
from models.vit import ViT
from training.lightning_module import LightningModule

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

NUM_CLASSES = 19
MODEL_IMG_SIZE = (1024, 1024)
PATCH_SIZE = 16
NUM_QUERIES = 100
NUM_BLOCKS = 3
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"

def build_transforms(img_height: int, img_width: int):
    input_t = Compose(
        [
            Resize((img_height, img_width), InterpolationMode.BILINEAR),
            ToTensor(),
        ]
    )
    target_t = Compose([Resize((img_height, img_width), InterpolationMode.NEAREST)])
    return input_t, target_t

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
        patch_size=PATCH_SIZE,
        backbone_name=BACKBONE_NAME,
    )
    network = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=NUM_QUERIES,
        num_blocks=NUM_BLOCKS,
        masked_attn_enabled=True,
    )

    lightningModule = LightningModule(
        network=network,
        img_size=MODEL_IMG_SIZE,
        num_classes=NUM_CLASSES,
        attn_mask_annealing_enabled=False,
        attn_mask_annealing_start_steps=None,
        attn_mask_annealing_end_steps=None,
        lr=1e-4,
        llrd=0.8,
        llrd_l2_enabled=True,
        lr_mult=1.0,
        weight_decay=0.05,
        poly_power=0.9,
        warmup_steps=(500, 1000),
        ckpt_path=ckpt_path,
        delta_weights=False,
        load_ckpt_class_head=True,
    )

    model = lightningModule.network
    model.to(device)
    model.eval()
    return model


def load_mask(path: str, target_transform):
    mask = Image.open(path).convert("L")
    mask = target_transform(mask)
    ood_gts = np.array(mask)
    if ood_gts.ndim == 3:
        ood_gts = ood_gts[..., 0]
    if "RoadAnomaly" in path:
        ood_gts = np.where((ood_gts == 2), 1, ood_gts)
    if "LostAndFound" in path:
        ood_gts = np.where((ood_gts == 0), 255, ood_gts)
        ood_gts = np.where((ood_gts == 1), 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
    if "Streethazard" in path:
        ood_gts = np.where((ood_gts == 14), 255, ood_gts)
        ood_gts = np.where((ood_gts < 20), 0, ood_gts)
        ood_gts = np.where((ood_gts == 255), 1, ood_gts)

    return ood_gts

def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0
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

    input_transform, target_transform = build_transforms(MODEL_IMG_SIZE[0], MODEL_IMG_SIZE[1])

    model = build_model(args.ckpt, device)
    print("[OK] EoMT loaded correctly")

    anomaly_score_list = []
    ood_gts_list = []

    image_paths = sorted(glob.glob(args.input))

    if len(image_paths) == 0:
        raise RuntimeError("No input images found")

    print(f"Processing {len(image_paths)} images...")

    for idx, img_path in enumerate(image_paths):
        print(img_path)
        img = input_transform(Image.open(img_path).convert("RGB"))
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            mask_logits_list, class_logits_list = model(img)

        mask_logits = mask_logits_list[-1]
        class_logits = class_logits_list[-1]

        mask_logits = F.interpolate(
            mask_logits,
            size=MODEL_IMG_SIZE,
            mode="bilinear",
            align_corners=False,
        )

        per_pixel_logits = LightningModule.to_per_pixel_logits_semantic(
                mask_logits, class_logits
            )

        per_pixel_logits = per_pixel_logits[0]
        logits = per_pixel_logits / args.temp
        per_pixel_probs = F.softmax(logits, dim=0)

        if args.method == "msp":
            score = 1.0 - torch.max(per_pixel_probs, dim=0).values
            score = score.cpu().numpy()

        elif args.method == "maxlogit":
            score = -torch.max(logits, dim=0).values
            score = score.cpu().numpy()

        elif args.method == "maxentropy":
            probs = per_pixel_probs.clamp(min=1e-12)
            score = -(probs * probs.log()).sum(dim=0)
            score = score.cpu().numpy()

        elif args.method == "rba":
            mask_probs = mask_logits.sigmoid()
            class_probs = (class_logits / args.temp).softmax(dim=-1)[..., :-1]
            max_class_probs = class_probs.max(dim=-1)[0]
            query_scores = mask_probs * max_class_probs[:, :, None, None]
            max_query_score = query_scores.max(dim=1)[0]
            score = (1.0 - max_query_score).squeeze(0).cpu().numpy()

        pathGT = img_path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        if not os.path.exists(pathGT):
            print(f"[WARNING] GT not found: {pathGT}")
            continue

        ood_gts = load_mask(pathGT, target_transform)

        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(score)

        if args.method == "maxlogit":
            del per_pixel_logits
        del mask_logits, class_logits
        if device == "cuda":
            torch.cuda.empty_cache()

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(image_paths)} images")

    if not anomaly_score_list:
        print("No OOD pixels found in provided dataset.")
        return

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = ood_gts == 1
    ind_mask = ood_gts == 0

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    scores = np.concatenate((ind_out, ood_out))
    labels = np.concatenate((ind_label, ood_label))

    auprc = average_precision_score(labels, scores)
    fpr95 = fpr_at_95_tpr(scores, labels)

    print(f"\n=== RESULTS ({args.method.upper()}) ===")
    print(f"AUPRC: {auprc * 100:.2f}")
    print(f"FPR@95TPR: {fpr95 * 100:.2f}")

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')
    file.write("DATASET  " + str(args.input) + "\n")
    file.write("METHOD  " + str(args.method) + "\n")
    file.write("TEMP    " + str(args.temp) + "\n")
    file.write(('    AUPRC score:' + str(auprc*100.0) + '   FPR@TPR95:' + str(fpr95*100.0) ))
    file.write( "\n")
    file.close()

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
