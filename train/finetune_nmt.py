import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# Resolve paths
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TRAIN_DIR)


if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import components from load_eegpt.py
from load_eegpt import NMTDataset, load_eegpt_model

if __name__ == "__main__":
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoints", "eegpt_mcae_58chs_4s_large4E.ckpt")
    CSV_PATH = os.path.join(ROOT_DIR, "data", "processed_windows_eegpt.csv")
    SAVE_DIR = os.path.join(ROOT_DIR, "checkpoints", "finetuned")
    os.makedirs(SAVE_DIR, exist_ok=True)

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

    # Load Model using function from load_eegpt.py
    model = load_eegpt_model(
        checkpoint_path=CHECKPOINT_PATH,
        num_classes=2,
        in_channels=len(nmt_21_channels),
        use_channels_names=eegpt_19_channels,
        desired_time_len=1024,
        use_chan_conv=True,
        device=device
    )

    # Dataset & Dataloader Setup
    print(f"\n--> Loading dataset from: {CSV_PATH}", flush=True)
    if os.path.exists(CSV_PATH):
        train_dataset = NMTDataset(CSV_PATH, split="train")
        eval_dataset = NMTDataset(CSV_PATH, split="evaluation")
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
        eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
        print(f"    Train Samples: {len(train_dataset)} | Eval Samples: {len(eval_dataset)}", flush=True)
        
        # Class Balance Handler: Compute class weights from training labels
        train_labels = train_dataset.data['label'].values
        computed_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights_tensor = torch.tensor(computed_weights, dtype=torch.float32).to(device)
        
        # Training Loop Configuration
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        
        EPOCHS = 5
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = GradScaler('cuda')
        
        best_eval_acc = 0.0
        patience = 2
        patience_counter = 0

        print("\n--> Starting Fine-Tuning Loop...", flush=True)
        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            print(f"\n--- Epoch {epoch}/{EPOCHS} Started ---", flush=True)
            
            # Progress bar with tqdm for training loop
            loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]", leave=True, dynamic_ncols=True)
            for step, (x, y) in enumerate(loop):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                with autocast('cuda'):
                    logits = model(x)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=-1)
                train_correct += (preds == y).sum().item()
                train_total += y.size(0)

                current_acc = (train_correct / train_total) * 100
                loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%")

            # Step the learning rate scheduler at the end of each epoch
            scheduler.step()

            train_acc = train_correct / train_total
            print(f"--> Epoch {epoch} Complete | Train Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc*100:.2f}%", flush=True)

            # Validation Loop with Metrics Collection
            model.eval()
            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for x, y in eval_loader:
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

            # Calculate validation metrics
            eval_acc = (all_preds == all_labels).mean()
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            try:
                tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            except ValueError:
                sensitivity, specificity = 0.0, 0.0

            try:
                auroc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auroc = 0.0

            print(f"           --> Validation Metrics:", flush=True)
            print(f"               Acc: {eval_acc*100:.2f}% | F1: {f1:.4f} | Sens: {sensitivity:.4f} | Spec: {specificity:.4f} | AUROC: {auroc:.4f}", flush=True)

            # Early Stopping and Checkpointing Logic
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                patience_counter = 0
                save_path = os.path.join(SAVE_DIR, "best_eegpt_model.pt")
                torch.save(model.state_dict(), save_path)
                print(f"           💾 Saved new best model checkpoint to {save_path}", flush=True)
            else:
                patience_counter += 1
                print(f"           ⚠️ Validation accuracy did not improve. Patience: {patience_counter}/{patience}", flush=True)
                if patience_counter >= patience:
                    print(f"           🛑 Early stopping triggered at epoch {epoch}!", flush=True)
                    break
    else:
        print(f"\n⚠️ Notice: Could not find {CSV_PATH}. Skipping training loop.", flush=True) 