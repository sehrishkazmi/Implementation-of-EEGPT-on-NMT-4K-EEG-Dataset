import os
import sys
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, 
    f1_score, 
    roc_auc_score, 
    roc_curve
)

# Resolve paths for the updated directory structure (e:\EEGPT-NMT\test\)
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))          # e:\EEGPT-NMT\test
ROOT_DIR = os.path.dirname(EVAL_DIR)                        # e:\EEGPT-NMT
TRAIN_DIR = os.path.join(ROOT_DIR, "train")                 # e:\EEGPT-NMT\train

# Ensure both root and train directories are in sys.path
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if TRAIN_DIR not in sys.path:
    sys.path.append(TRAIN_DIR)

# Import components from load_eegpt.py (located in train/)
from train.load_eegpt import NMTDataset, load_eegpt_model

if __name__ == "__main__":
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoints", "finetuned", "best_eegpt_model.pt")
    CSV_PATH = os.path.join(ROOT_DIR, "data", "processed_windows_eegpt.csv")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "evaluation_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    nmt_21_channels = [
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
        "T7", "C3", "CZ", "C4", "T8",
        "P7", "P3", "PZ", "P4", "P8",
        "O1", "O2", "A1", "A2"
    ]
    eegpt_19_channels = [
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
        "T7", "C3", "CZ", "C4", "T8",
        "P7", "P3", "PZ", "P4", "P8",
        "O1", "O2"
    ]

    # 1. Load Model Architecture and Best Weights
    print(f"\n--> Loading model architecture and weights from: {CHECKPOINT_PATH}", flush=True)
    model = load_eegpt_model(
        checkpoint_path=os.path.join(ROOT_DIR, "checkpoints", "eegpt_mcae_58chs_4s_large4E.ckpt"),
        num_classes=2,
        in_channels=len(nmt_21_channels),
        use_channels_names=eegpt_19_channels,
        desired_time_len=1024,
        use_chan_conv=True,
        device=device
    )

    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    else:
        raise FileNotFoundError(f"Could not find best model checkpoint at {CHECKPOINT_PATH}")

    model.eval()

    # 2. Load Evaluation Dataset
    print(f"--> Loading evaluation dataset from: {CSV_PATH}", flush=True)
    eval_dataset = NMTDataset(CSV_PATH, split="evaluation")
    
    # Using num_workers=0 to prevent Windows multiprocessing hangs
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    print(f"    Evaluation Samples: {len(eval_dataset)}", flush=True)

    # 3. Run Inference with Progress Bar
    print("\n--> Running evaluation inference...", flush=True)
    all_preds = []
    all_labels = []
    all_probs = []

    inference_loop = tqdm(eval_loader, desc="Evaluating", leave=True, dynamic_ncols=True)
    with torch.no_grad():
        for x, y in inference_loop:
            x, y = x.to(device), y.to(device)
            with autocast('cuda'):
                logits = model(x)
            
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 4. Compute Standard Metrics (at threshold = 0.5)
    eval_acc = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auroc = roc_auc_score(all_labels, all_probs)

    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS (Threshold = 0.5):")
    print("="*50)
    print(f"Accuracy    : {eval_acc*100:.2f}%")
    print(f"Macro F1    : {f1:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"AUROC       : {auroc:.4f}")
    print("="*50)

    # 5. Plot and Save Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Normal', 'Abnormal'], 
                yticklabels=['Normal', 'Abnormal'])
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title('Confusion Matrix (Best Model)', fontweight='bold')
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved Confusion Matrix plot to {cm_path}", flush=True)

    # 6. Plot and Save ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC)', fontweight='bold')
    plt.legend(loc="lower right")
    roc_path = os.path.join(OUTPUT_DIR, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved ROC Curve plot to {roc_path}", flush=True)

    # 7. Threshold Tuning for Clinical Safety
    print("\n--> Performing Threshold Tuning to Optimize Sensitivity...", flush=True)
    best_thresh = 0.5
    best_target_score = 0.0

    # Sweep thresholds from 0.1 to 0.9
    for th in np.linspace(0.1, 0.9, 81):
        preds_th = (all_probs >= th).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(all_labels, preds_th).ravel()
        sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
        f1_t = f1_score(all_labels, preds_th, average='macro', zero_division=0)
        
        score = (2 * sens_t * f1_t) / (sens_t + f1_t) if (sens_t + f1_t) > 0 else 0.0

        if score > best_target_score:
            best_target_score = score
            best_thresh = th

    print(f"    Optimal Threshold found: {best_thresh:.2f}")
    
    # Evaluate at optimal threshold
    optimal_preds = (all_probs >= best_thresh).astype(int)
    opt_cm = confusion_matrix(all_labels, optimal_preds)
    opt_tn, opt_fp, opt_fn, opt_tp = opt_cm.ravel()
    opt_sens = opt_tp / (opt_tp + opt_fn)
    opt_spec = opt_tn / (opt_tn + opt_fp)
    opt_f1 = f1_score(all_labels, optimal_preds, average='macro', zero_division=0)
    opt_acc = (optimal_preds == all_labels).mean()

    print("\n" + "="*50)
    print(f"METRICS AT OPTIMAL THRESHOLD ({best_thresh:.2f}):")
    print("="*50)
    print(f"Accuracy    : {opt_acc*100:.2f}%")
    print(f"Macro F1    : {opt_f1:.4f}")
    print(f"Sensitivity : {opt_sens:.4f}")
    print(f"Specificity : {opt_spec:.4f}")
    print("="*50)
    print(f"\nAll evaluation outputs successfully saved in: {OUTPUT_DIR}")