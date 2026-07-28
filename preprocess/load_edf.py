from pathlib import Path
import mne
import pandas as pd
import re

# Set path to the ROOT directory containing both 'train' and 'evaluation'
EDF_PATH = r"E:\Dataset\NMT-4K-EEG"  

# Turn off verbose logging from MNE when reading headers, MNE normally prints info logs every time it touches an EDF file. 
# Setting this to "ERROR" keeps the terminal output clean.
mne.set_log_level("ERROR")

# The exact 20 standard channels required by eegpt-encoder embeddings.
TARGET_CHANNELS = {
    "FP1", "FPZ", "FP2", "F7", "F3", "FZ", "F4", "F8", 
    "T7", "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", 
    "P4", "P8", "O1", "O2"
}

# NMT uses 1958 conventions; EEGPT uses 10-10. This maps them perfectly.
# efines a mapping dictionary to convert legacy 1958 10–20 channel names (T3, T4, T5, T6) found in NMT-4K-EEG to modern 10–10 names (T7, T8, P7, P8) required by EEGPT.
ALIASES = {
    "T3": "T7", "T4": "T8", 
    "T5": "P7", "T6": "P8",
}

# re.sub(r'(?i)^eeg\s*', '', name): Case-insensitively ((?i)) strips any leading "EEG " or "EEG" prefix.re.sub(...): Strips reference extensions like -REF, -AVG, -LE, or -M2 from the end of the channel string.
# .strip() removes remaining surrounding spaces.
# cleaned_upper = cleaned.upper(): Standardizes the string to uppercase (e.g., fp1 $\rightarrow$ FP1)
# .ALIASES.get(...): Looks up the channel in ALIASES. If cleaned_upper is "T3", it returns "T7". 
# If it's not in the dictionary (like "C3"), it returns "C3" unchanged.
def clean_channel_name(name: str) -> str:
    """Normalize raw channel labels into EEGPT 10-10 uppercase format."""
    cleaned = re.sub(r'(?i)^eeg\s*', '', name)
    cleaned = re.sub(r'(?i)[-_\s](REF|AVG|LE|M2|M1|A1|A2)$', '', cleaned).strip()
    cleaned_upper = cleaned.upper()
    return ALIASES.get(cleaned_upper, cleaned_upper)

# file_path.parts: Breaks a path into individual directory names (e.g., ["E:", "Dataset", "NMT-4K-EEG", "train", "normal", "file.edf"]) and converts them all to lowercase.
def extract_path_metadata(file_path: Path) -> tuple[str, str]:
    """Parse split (train/eval) and label (normal/abnormal) from folder structure."""
    parts = [p.lower() for p in file_path.parts]
    
    # Extract split, Inspects directory names in the path to automatically tag whether the file belongs to the train or evaluation set.
    if "train" in parts:
        split = "train"
    elif "evaluation" in parts or "eval" in parts or "test" in parts:
        split = "evaluation"
    else:
        split = "other"
        
    # Extract label
    if "abnormal" in parts:
        label = "abnormal"
    elif "normal" in parts:
        label = "normal"
    else:
        label = "unknown"
        
    return split, label

# Gets split and label tags for the current file using the helper function.
def inspect_edf_file(file_path: Path) -> dict:
    """Extract metadata from a single EDF file without loading heavy data arrays."""
    split, label = extract_path_metadata(file_path)
    
    try:
        # preload=False: Reads only the header metadata (channel names, sampling rates, time length) without reading the heavy signal arrays into RAM. 
        # This makes checking ~4,500 files run in seconds rather than minutes.
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose="ERROR")


        # Extracts raw channel strings and runs every channel through our cleaning and mapping function.
        orig_channels = raw.ch_names
        cleaned_channels = [clean_channel_name(ch) for ch in orig_channels]

        # .intersection(...): Finds which channels in this file exist inside TARGET_CHANNELS. 
        # (Set Difference): Identifies which target channels are missing from this file.
        matched_channels = set(cleaned_channels).intersection(TARGET_CHANNELS)
        missing_channels = TARGET_CHANNELS - set(cleaned_channels)

        # Calculates recording duration in seconds by dividing total time samples (n_times) by sampling frequency (sfreq).
        duration_sec = raw.n_times / raw.info["sfreq"]

        # Packages file metadata into a dictionary. 
        # has_min_eegpt_channels flags True if 19 target channels are found (since NMT natively lacks FPZ).
        return {
            "file_name": file_path.name,
            "split": split,
            "label": label,
            "path": str(file_path),
            "sampling_rate_hz": raw.info["sfreq"],
            "total_channels": len(orig_channels),
            "duration_sec": round(duration_sec, 2),
            "duration_min": round(duration_sec / 60, 2),
            "target_20_matched": len(matched_channels),
            "has_all_20_channels": len(missing_channels) == 0,
            "has_min_eegpt_channels": len(matched_channels) >= 19, 
            "missing_target_channels": list(missing_channels) if missing_channels else "None",
            "all_channel_names": orig_channels
        }
    except Exception as e:
        return {
            # Error Catching: If an EDF file is corrupted or unreadable, the script catches the error and logs it in the report rather than crashing the pipeline.
            "file_name": file_path.name,
            "split": split,
            "label": label,
            "path": str(file_path),
            "error": str(e)
        }

# rglob("*"): Recursively traverses all subdirectories under E:\Dataset\NMT-4K-EEG. 
# Filtering by .is_file() and .suffix.lower() == ".edf" gathers every EDF file across train/normal, train/abnormal, evaluation/normal, and evaluation/abnormal.
def analyze_directory(data_dir: str):
    path = Path(data_dir)
    
    if path.is_file() and path.suffix.lower() == ".edf":
        edf_files = [path]
    elif path.is_dir():
        # SAFELY grab all files across ALL subdirectories and filter by extension
        edf_files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".edf"]
    else:
        print(f"Error: Provided path '{data_dir}' is neither a valid file nor directory.")
        return

    if not edf_files:
        print(f"No .edf files found in '{data_dir}'.")
        return

    print(f"\n🔍 Scanning {len(edf_files)} EDF file(s) across dataset hierarchy...\n")

    
    results = [inspect_edf_file(f) for f in edf_files]

    # Processes all EDF files into a list of dictionaries, converts them into a Pandas DataFrame, and splits valid readings from errored ones.
    df = pd.DataFrame(results)
    
    valid_df = df[df["error"].isna()] if "error" in df.columns else df
    error_df = df[df["error"].notna()] if "error" in df.columns else pd.DataFrame()

    # Computes and prints breakdown tables in terminal: total files, splits, normal vs. abnormal counts, sampling frequencies, channel matching metrics, and duration statistics.
    print("=" * 60)
    print(" DATASET SUMMARY METRICS")
    print("=" * 60)
    print(f"Total EDF Files Found:        {len(df)}")
    print(f"Successfully Read:            {len(valid_df)}")
    print(f"Failed / Unreadable Files:    {len(error_df)}")
    
    if not valid_df.empty:
        print("\nFiles Breakdown by Split:")
        print(valid_df["split"].value_counts().to_string())

        print("\nFiles Breakdown by Class Label:")
        print(valid_df["label"].value_counts().to_string())

        print("\nFiles Breakdown by Split & Label:")
        print(pd.crosstab(valid_df["split"], valid_df["label"]).to_string())

        print("\nSampling Rates Breakdown:")
        print(valid_df["sampling_rate_hz"].value_counts().to_string())

        print("\nFiles with >= 19 Matched EEGPT Channels:")
        print(valid_df["has_min_eegpt_channels"].value_counts().to_string())
        
        print(f"\nAverage Duration:             {valid_df['duration_min'].mean():.2f} minutes")
        print(f"Shortest File:                {valid_df['duration_min'].min():.2f} minutes")
        print(f"Longest File:                 {valid_df['duration_min'].max():.2f} minutes")

    print("\n" + "=" * 60)
    print(" FIRST 5 FILES PREVIEW")
    print("=" * 60)
    preview_cols = ["file_name", "split", "label", "sampling_rate_hz", "target_20_matched", "duration_min"]
    cols_to_show = [c for c in preview_cols if c in valid_df.columns]
    print(valid_df[cols_to_show].head().to_string(index=False))
    
    # Get the directory where this script lives (the 'preprocess' folder)
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()

    # Save CSV directly in the 'preprocess' directory
    output_csv = script_dir / "eegpt_full_dataset_inspection.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n Full report exported to '{output_csv}'")

if __name__ == "__main__":
    analyze_directory(EDF_PATH)