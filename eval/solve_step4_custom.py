# solve_step4_final.py
import os
import glob
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn.functional as F
from sklearn.metrics import roc_curve, average_precision_score

# --- CONFIGURATION ---
NUM_CLASSES = 20
try:
    from erfnet import ERFNet
except ImportError:
    import sys
    sys.path.append('.')
    from erfnet import ERFNet

input_transform = Compose([Resize((512, 1024), Image.BILINEAR), ToTensor()])
target_transform = Compose([Resize((512, 1024), Image.NEAREST)])

def get_anomaly_score(logits, method, temperature=1.0):
    logits = logits / temperature
    if method == "MSP":
        probs = F.softmax(logits, dim=1)
        confidence, _ = torch.max(probs, dim=1)
        return 1.0 - confidence.squeeze(0).cpu().numpy()
    elif method == "MaxLogit":
        confidence, _ = torch.max(logits, dim=1)
        return -confidence.squeeze(0).cpu().numpy()
    elif method == "MaxEntropy":
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy.squeeze(0).cpu().numpy()
    return np.zeros((512, 1024))

def calc_metrics(anomaly_scores, ground_truths):
    preds = np.concatenate(anomaly_scores)
    gts = np.concatenate(ground_truths)
    auprc = average_precision_score(gts, preds)
    fpr, tpr, _ = roc_curve(gts, preds)
    if len(tpr) > 0 and np.max(tpr) >= 0.95:
        idx = np.argmax(tpr >= 0.95)
        fpr95 = fpr[idx]
    else:
        fpr95 = 0.0
    return auprc, fpr95

def main():
    parser = ArgumentParser()
    # Point this to the 'frames' folder inside RoadAnomaly
    parser.add_argument("--dataroot", default="../data/RoadAnomaly_jpg/frames")
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    args = parser.parse_args()

    device = torch.device('cpu')
    print("--- SOLVING STEP 4: PIXEL-BASED BASELINES ---")

    # 1. LOAD MODEL
    print(f"Loading Model: {args.loadWeights}")
    model = ERFNet(NUM_CLASSES)
    modelpath = os.path.join(args.loadDir, args.loadWeights)
    checkpoint = torch.load(modelpath, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    # 2. MATCH IMAGES WITH LABELS
    print(f"Scanning {args.dataroot}...")
    valid_pairs = []
    
    # Get all JPGs in the folder
    jpg_files = glob.glob(os.path.join(args.dataroot, "*.jpg"))
    
    for jpg_path in jpg_files:
        # Construct expected label path
        # Logic: animals01.jpg -> animals01.labels/labels_semantic.png
        basename = os.path.basename(jpg_path) # animals01.jpg
        folder_name = basename.replace(".jpg", ".labels") # animals01.labels
        
        label_path = os.path.join(args.dataroot, folder_name, "labels_semantic.png")
        
        if os.path.exists(label_path):
            valid_pairs.append((jpg_path, label_path))
    
    print(f"Found {len(valid_pairs)} valid Image-Label pairs.")
    
    if len(valid_pairs) == 0:
        print("ERROR: Still found 0 pairs. Please ensure 'frames' folder contains both JPGs and .labels folders.")
        return

    # 3. PROCESSING
    results = {"MSP": [], "MaxLogit": [], "MaxEntropy": [], "MSP_T0.5": [], "MSP_T0.75": [], "MSP_T1.1": []}
    ground_truths = []

    # Process first 20 images (Remove [:20] to run ALL images)
    #for i, (img_path, label_path) in enumerate(valid_pairs[:20]):
    #now all images
    for i, (img_path, label_path) in enumerate(valid_pairs):
        print(f"[{i+1}] Processing: {os.path.basename(img_path)}")

        # Input
        image = input_transform(Image.open(img_path).convert('RGB')).unsqueeze(0).float().to(device)
        
        # Ground Truth (2=Anomaly, 1=Road, 0=Background)
        gt_img = Image.open(label_path)
        gt_mask = np.array(target_transform(gt_img))
        binary_gt = np.where(gt_mask == 2, 1, 0).flatten() # 1 if Anomaly, else 0
        ground_truths.append(binary_gt)

        # Inference
        with torch.no_grad():
            logits = model(image)

        # Metrics
        results["MSP"].append(get_anomaly_score(logits, "MSP").flatten())
        results["MaxLogit"].append(get_anomaly_score(logits, "MaxLogit").flatten())
        results["MaxEntropy"].append(get_anomaly_score(logits, "MaxEntropy").flatten())
        
        # Temperature Scaling
        results["MSP_T0.5"].append(get_anomaly_score(logits, "MSP", 0.5).flatten())
        results["MSP_T0.75"].append(get_anomaly_score(logits, "MSP", 0.75).flatten())
        results["MSP_T1.1"].append(get_anomaly_score(logits, "MSP", 1.1).flatten())

    # 4. PRINT TABLE
    print("\n" + "="*50)
    print(f"{'METHOD':<15} | {'AUPRC (Accuracy)':<20} | {'FPR95 (Error Rate)':<20}")
    print("-" * 55)
    
    for method in results:
        auprc, fpr95 = calc_metrics(results[method], ground_truths)
        print(f"{method:<15} | {auprc:.4f}               | {fpr95:.4f}")
    
    print("="*50)

if __name__ == '__main__':
    main()