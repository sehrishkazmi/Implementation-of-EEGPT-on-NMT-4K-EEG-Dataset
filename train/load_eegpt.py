import os
import sys
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Resolve paths: TRAIN_DIR is EEGPT-NMT/train, ROOT_DIR is EEGPT-NMT
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TRAIN_DIR)

# Add root directory to sys.path so 'model' package is discoverable
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import the modified EEGPTClassifier model architecture
from model.EEGPT_mcae_finetune_modified import EEGPTClassifier


# ==========================================
# DATASET DEFINITION
# ==========================================
class NMTDataset(Dataset):
    def __init__(self, csv_file: str, split: str = "train"):
        df = pd.read_csv(csv_file)
        self.data = df[df['split'] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pt_path = row['tensor_path']
        window_idx = row['window_index']
        
        # Resolve absolute path if stored relative
        if not os.path.isabs(pt_path):
            full_pt_path = os.path.join(ROOT_DIR, pt_path)
        else:
            full_pt_path = pt_path

        tensors = torch.load(full_pt_path, map_location="cpu", weights_only=True)
        x = tensors[window_idx] if tensors.dim() > 2 else tensors
        
        # Ensure tensor is float32 to match newly initialized layers (chan_conv / head)
        x = x.to(torch.float32)
        
        y = torch.tensor(row['label'], dtype=torch.long)
        
        return x, y


# ==========================================
# MODEL LOADER FUNCTION
# ==========================================
def load_eegpt_model(
    checkpoint_path: str,
    num_classes: int = 2,
    in_channels: int = 21,
    use_channels_names: list = None,
    desired_time_len: int = 1024,
    use_chan_conv: bool = True,
    device: torch.device = None
):
    """
    Instantiates EEGPTClassifier for the 21-channel NMT dataset and loads pretrained weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {os.path.abspath(checkpoint_path)}")

    print("--> Initializing EEGPTClassifier model...")
    
    if use_channels_names is None:
        use_channels_names = [
            'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
            'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2'
        ]

    model = EEGPTClassifier(
        num_classes=num_classes,
        in_channels=in_channels,
        img_size=[len(use_channels_names), desired_time_len],
        use_channels_names=use_channels_names,
        desired_time_len=desired_time_len,
        use_chan_conv=use_chan_conv,
        use_predictor=False
    )

    print(f"--> Loading state dict from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state_dict = model.state_dict()
    matched_state_dict = {}

    for k, v in state_dict.items():
        clean_key = k.replace("model.", "")
        if clean_key in model_state_dict and model_state_dict[clean_key].shape == v.shape:
            matched_state_dict[clean_key] = v
        elif k in model_state_dict and model_state_dict[k].shape == v.shape:
            matched_state_dict[k] = v

    missing_keys, _ = model.load_state_dict(matched_state_dict, strict=False)
    
    print(f"--> Successfully loaded {len(matched_state_dict)} weight tensors.")
    if missing_keys:
        print(f"    Notice: {len(missing_keys)} uninitialized layers (e.g., channel conv / classifier head).")

    model.to(device)
    return model


# ==========================================
# VERIFICATION BLOCK (Using Actual .pt Files)
# ==========================================
if __name__ == "__main__":
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoints", "eegpt_mcae_58chs_4s_large4E.ckpt")
    CSV_PATH = os.path.join(ROOT_DIR, "data", "processed_windows_eegpt.csv")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    model = load_eegpt_model(
        checkpoint_path=CHECKPOINT_PATH,
        num_classes=2,
        in_channels=len(nmt_21_channels),
        use_channels_names=eegpt_19_channels,
        desired_time_len=1024,
        use_chan_conv=True,
        device=device
    )

    model.eval()
    
    if os.path.exists(CSV_PATH):
        print(f"\n--> Loading actual data sample from: {CSV_PATH}")
        dataset = NMTDataset(CSV_PATH, split="train")
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Grab a real batch of .pt windows
        x_real, y_real = next(iter(loader))
        x_real = x_real.to(device)
        
        print("\nExecuting forward pass verification with actual .pt dataset windows...")
        with torch.no_grad():
            output = model(x_real)

        print(f"Input Shape:  {x_real.shape}  -> [Batch, Channels, Time]")
        print(f"Output Shape: {output.shape}        -> [Batch, Num_Classes]")
        print(f"True Labels:  {y_real.tolist()}")
        print("\nActual data forward pass verification successful!")
    else:
        print(f"\n⚠️ Warning: CSV file not found at {CSV_PATH}. Could not run verification with actual .pt files.")