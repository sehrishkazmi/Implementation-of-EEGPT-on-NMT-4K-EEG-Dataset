import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import mne
from tqdm import tqdm

mne.set_log_level("ERROR")

# All 21 EEG channels from NMT-4k-EEG (19 standard 10-20 channels + A1, A2)
TARGET_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", 
    "T7", "C3", "CZ", "C4", "T8", 
    "P7", "P3", "PZ", "P4", "P8", 
    "O1", "O2", "A1", "A2"
]

# Standard renaming map for 10-20 to ACNS 10-10 conventions and reference aliases
ALIASES = {
    "T3": "T7", "T4": "T8",
    "T5": "P7", "T6": "P8",
    "M1": "A1", "M2": "A2"
}

def clean_channel_name(name: str) -> str:
    """Normalize raw channel labels into uppercase standard channel names."""
    cleaned = re.sub(r'(?i)^eeg\s*', '', name)
    cleaned = re.sub(r'(?i)[-_\s](REF|AVG|LE)$', '', cleaned).strip()
    cleaned_upper = cleaned.upper()
    return ALIASES.get(cleaned_upper, cleaned_upper)

def preprocess_single_edf_eegpt(file_path: str, target_sfreq: float = 256.0, window_sec: float = 10.0):
    """
    Preprocesses a single EDF file from NMT-4k-EEG:
    1. Standardizes channel names (maps legacy names & cleans prefixes/suffixes).
    2. Extracts all 21 EEG channels (excluding ECG).
    3. Clips extreme amplitude artifacts [-800uV, +800uV].
    4. Applies Common Average Reference (CAR).
    5. Resamples to 256 Hz.
    6. Slices recording into non-overlapping 10-second windows (2560 time points).
    7. Converts Volts to uV.
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")

        # 1. Standardize channel names
        rename_dict = {orig: clean_channel_name(orig) for orig in raw.ch_names}
        raw.rename_channels(rename_dict)

        # 2. Check for missing channels
        missing = set(TARGET_CHANNELS) - set(raw.ch_names)
        if missing:
            return None, f"Missing required channels: {missing}"

        # Reorder channels strictly to TARGET_CHANNELS order
        raw.pick_channels(TARGET_CHANNELS, ordered=True)

        # 3. Minimum duration check
        total_duration = raw.times[-1]  
        if total_duration < window_sec:  
            return None, f"Recording too short ({total_duration:.1f}s < {window_sec}s)"

        # Crop to maximum recording length (21 minutes / 1260 seconds max)
        tmax = min(1260.0, raw.times[-1])
        raw.crop(tmin=0.0, tmax=tmax, include_tmax=False)

        # 4. Clip extreme amplitude artifacts and fill NaNs
        def safe_clip_and_clean(x):
            cleaned_x = pd.to_numeric(pd.Series(x), errors='coerce').fillna(0.0).values
            return np.clip(cleaned_x, -800e-6, 800e-6)

        raw.apply_function(safe_clip_and_clean)

        # 5. Apply Common Average Reference (CAR)
        raw.set_eeg_reference(ref_channels="average", projection=False)
        
        # 6. Resample signal to 256 Hz
        raw.resample(sfreq=target_sfreq)

        data = raw.get_data()
        
        # 7. Window Slicing (10s @ 256Hz = 2,560 time points)
        window_size = int(target_sfreq * window_sec) 
        num_windows = data.shape[1] // window_size

        if num_windows == 0:
            return None, f"No full {window_sec}s windows available after resampling"

        windows = []
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            win_data = data[:, start:end] * 1e6  # Convert V to uV
            windows.append(win_data)

        return np.array(windows, dtype=np.float16), None

    except Exception as e:
        return None, str(e)

def run_pipeline(metadata_csv: str = "data/metadata.csv", output_dir: str = "data/processed_eegpt"):
    """
    Runs full preprocessing pipeline on all files listed in metadata_csv.
    Generates tensor files (.pt) of shape [N, 21, 2560] and a window-level metadata index.
    """
    df = pd.read_csv(metadata_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_records = []
    failed_count = 0
    first_error = None

    print(f"🚀 Running Preprocessing (21 Channels @ 256 Hz) on {len(df)} recordings...\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Recordings"):
        edf_file = row["file_path"] if "file_path" in row else row["path"]

        if not os.path.exists(str(edf_file)):
            if first_error is None:
                first_error = f"File not found: '{edf_file}'"
            failed_count += 1
            continue

        rec_id = row.get("recording_id", Path(edf_file).stem)
        windows, err = preprocess_single_edf_eegpt(str(edf_file), target_sfreq=256.0, window_sec=10.0)

        if windows is None:
            if first_error is None and err:
                first_error = f"Processing error on {rec_id}: {err}"
            failed_count += 1
            continue

        # Save PyTorch tensor to disk: Shape [num_windows, 21, 2560]
        save_path = output_path / f"{rec_id}.pt"
        torch.save(torch.from_numpy(windows), save_path)

        # Map each individual window for DataLoader indexing
        for win_idx in range(len(windows)):
            rec = {
                "window_id": f"{rec_id}_w{win_idx}",
                "recording_id": rec_id,
                "patient_id": row.get("patient_id", rec_id.split("_")[0]),
                "window_index": win_idx,
                "tensor_path": str(save_path),
                "label": row["label"],
                "split": row.get("split", "train"),
            }
            if "age_years" in row:
                rec["age_years"] = row["age_years"]
            if "gender" in row:
                rec["gender"] = row["gender"]

            processed_records.append(rec)

    # Save index CSV
    windows_df = pd.DataFrame(processed_records)
    windows_csv_path = Path("data/processed_windows_eegpt.csv")
    windows_df.to_csv(windows_csv_path, index=False)

    print("\n" + "=" * 60)
    print(" EEGPT PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total EDF Recordings Processed:  {len(df)}")
    print(f"Successfully Saved (.pt):       {len(df) - failed_count}")
    print(f"Failed / Skipped Recordings:    {failed_count}")
    print(f"Total Windows Sliced:           {len(windows_df)}")

    if first_error:
        print(f"\n⚠️ Sample error encountered: {first_error}")

if __name__ == "__main__":
    run_pipeline()