import os
import sys
import torch

# Resolve paths: TRAIN_DIR is EEGPT-NMT/train, ROOT_DIR is EEGPT-NMT
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TRAIN_DIR)

# Add root directory to sys.path so 'model' package is discoverable
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import the EEGPTClassifier model architecture
from model.EEGPT_mcae_finetune_modified import EEGPTClassifier


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
    Instantiates EEGPTClassifier for the 21-channel NMT dataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {os.path.abspath(checkpoint_path)}")

    print("--> Initializing EEGPTClassifier model...")
    
    # 19 standard 10-20 positional channels recognized by EEGPT's pre-trained dictionary
    if use_channels_names is None:
        use_channels_names = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']

    # Instantiate classifier with 21 raw input channels and 1D channel convolution
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
    
    # Updated for PyTorch 2.6+ checkpoint compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract state_dict (handles Lightning vs standard PyTorch checkpoints)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Filter and match keys between checkpoint and EEGPTClassifier
    model_state_dict = model.state_dict()
    matched_state_dict = {}

    for k, v in state_dict.items():
        clean_key = k.replace("model.", "")
        if clean_key in model_state_dict and model_state_dict[clean_key].shape == v.shape:
            matched_state_dict[clean_key] = v
        elif k in model_state_dict and model_state_dict[k].shape == v.shape:
            matched_state_dict[k] = v

    # Load matched parameters into model
    missing_keys, _ = model.load_state_dict(matched_state_dict, strict=False)
    
    print(f"--> Successfully loaded {len(matched_state_dict)} weight tensors.")
    if missing_keys:
        print(f"    Notice: {len(missing_keys)} uninitialized layers (e.g., channel conv / classifier head).")

    model.to(device)
    return model


if __name__ == "__main__":
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoints", "eegpt_mcae_58chs_4s_large4E.ckpt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Full 21 NMT raw electrodes (including A1 & A2)
    nmt_21_channels = [
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
        "T7", "C3", "CZ", "C4", "T8",
        "P7", "P3", "PZ", "P4", "P8",
        "O1", "O2", "A1", "A2"
    ]

    # 19 standard 10-20 positional electrodes for EEGPT target mapping
    eegpt_19_channels = [
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
        "T7", "C3", "CZ", "C4", "T8",
        "P7", "P3", "PZ", "P4", "P8",
        "O1", "O2"
    ]

    # Load model configured for 21 raw input channels
    model = load_eegpt_model(
        checkpoint_path=CHECKPOINT_PATH,
        num_classes=2,
        in_channels=len(nmt_21_channels),      # 21 channels
        use_channels_names=eegpt_19_channels,  # 19 standard positional channels
        desired_time_len=1024,                  # 4s @ 256 Hz
        use_chan_conv=True,                     # Enables 21 -> 19 channel projection
        device=device
    )

    model.eval()

    # Input Tensor: [Batch Size: 2, Electrodes: 21, Samples: 1024]
    dummy_input = torch.randn(2, len(nmt_21_channels), 1024).to(device)

    print("\nExecuting forward pass verification with 21 input channels...")
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input Shape:  {dummy_input.shape}  -> [Batch, 21 Channels, Time]")
    print(f"Output Shape: {output.shape}        -> [Batch, Num_Classes]")
    print("\n21-Channel forward pass verified successfully!")