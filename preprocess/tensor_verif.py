from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

def validate_processed_eegpt_data(
    tensor_dir: str = "data/processed_eegpt",
    metadata_csv: str = "data/processed_windows_eegpt.csv",
    expected_channels: int = 21,
    expected_samples: int = 2560
):
    tensor_path = Path(tensor_dir)
    csv_path = Path(metadata_csv)

    if not tensor_path.exists():
        print(f"⚠️ Directory '{tensor_dir}' does not exist yet.")
        return

    pt_files = list(tensor_path.glob("*.pt"))
    if not pt_files:
        print(f"⚠️ No .pt files found in '{tensor_dir}' yet.")
        return

    print(f"🔍 Found {len(pt_files)} .pt tensor files on disk. Running integrity checks...\n")

    valid_count = 0
    corrupted_count = 0
    shape_mismatch_count = 0
    nan_inf_count = 0
    total_windows_found = 0

    # Dictionary to track window counts per tensor on disk
    disk_tensor_window_counts = {}

    for pt_file in tqdm(pt_files, desc="Validating Tensors"):
        try:
            tensor = torch.load(pt_file, map_location="cpu")
            
            # Verify 3D tensor shape [num_windows, channels, samples]
            if tensor.ndim != 3:
                shape_mismatch_count += 1
                print(f"⚠️ {pt_file.name}: Invalid dimension count ({tensor.ndim}D instead of 3D)")
                continue
            
            n_win, n_ch, n_smp = tensor.shape

            if n_ch != expected_channels or n_smp != expected_samples or n_win == 0:
                shape_mismatch_count += 1
                print(f"⚠️ {pt_file.name}: Unexpected shape {tuple(tensor.shape)} (expected [N, {expected_channels}, {expected_samples}])")
                continue

            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                nan_inf_count += 1
                print(f"⚠️ {pt_file.name}: Contains NaN or Inf values")
                continue

            valid_count += 1
            total_windows_found += n_win
            disk_tensor_window_counts[pt_file.resolve()] = n_win

        except Exception as e:
            corrupted_count += 1
            print(f"❌ {pt_file.name}: Failed to read file ({e})")

    # -------------------------------------------------------------------------
    # CSV INDEX & CROSS-RECONCILIATION CHECKS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" METADATA CSV & DISK RECONCILIATION")
    print("=" * 60)

    if not csv_path.exists():
        print(f"ℹ️ Metadata file '{metadata_csv}' not found. Reconciliation skipped.")
        return

    df = pd.read_csv(csv_path)
    print(f"Total Window Entries in CSV Index: {len(df)}")

    # Resolve paths for accurate set operations regardless of slashes
    disk_paths = set(disk_tensor_window_counts.keys())
    
    # Handle tensor_path resolution from CSV
    if "tensor_path" in df.columns:
        csv_tensor_paths = set(Path(p).resolve() for p in df["tensor_path"].unique())
    else:
        # Fallback to matching by recording_id if tensor_path isn't explicitly listed
        csv_tensor_paths = set((tensor_path / f"{rec_id}.pt").resolve() for rec_id in df["recording_id"].unique())

    # 1. Files on disk but NOT in CSV (Orphaned / Unindexed files)
    unindexed_files = disk_paths - csv_tensor_paths
    
    # 2. Files referenced in CSV but NOT on disk (Missing files)
    missing_files = csv_tensor_paths - disk_paths

    # 3. Window Count Mismatch Check (Ensures CSV row count matches tensor dim 0)
    window_count_mismatches = []
    if "tensor_path" in df.columns:
        csv_counts_per_file = df.groupby("tensor_path").size()
        for path_str, csv_win_count in csv_counts_per_file.items():
            resolved_p = Path(path_str).resolve()
            if resolved_p in disk_tensor_window_counts:
                actual_win_count = disk_tensor_window_counts[resolved_p]
                if actual_win_count != csv_win_count:
                    window_count_mismatches.append((resolved_p.name, actual_win_count, csv_win_count))

    # Print Reconciliation Findings
    print(f"Distinct Tensors in CSV Index:    {len(csv_tensor_paths)}")
    print(f"Distinct Tensors Verified on Disk: {len(disk_paths)}")
    print(f"Unindexed .pt Files (Disk only):  {len(unindexed_files)}")
    print(f"Missing .pt Files (CSV only):     {len(missing_files)}")
    print(f"Window Count Mismatches:          {len(window_count_mismatches)}")

    if unindexed_files:
        print("\n⚠️ Sample Unindexed .pt Files (On disk, missing in CSV):")
        for f in list(unindexed_files)[:5]:
            print(f"   - {f.name}")

    if missing_files:
        print("\n❌ Sample Missing .pt Files (In CSV, missing on disk):")
        for f in list(missing_files)[:5]:
            print(f"   - {f.name}")

    if window_count_mismatches:
        print("\n⚠️ Sample Window Count Mismatches (Disk Tensor N != CSV Rows):")
        for name, disk_cnt, csv_cnt in window_count_mismatches[:5]:
            print(f"   - {name}: Disk has {disk_cnt} windows, CSV has {csv_cnt} rows")

    # Split Distribution Summary
    if "split" in df.columns:
        print("\n" + "=" * 60)
        print(" DATASET SPLIT BREAKDOWN")
        print("=" * 60)
        print("Window Breakdown by Split:")
        print(df["split"].value_counts().to_string())

        if "label" in df.columns:
            print("\nWindow Breakdown by Split & Class Label:")
            print(pd.crosstab(df["split"], df["label"], margins=True).to_string())

if __name__ == "__main__":
    validate_processed_eegpt_data()

# another script to check the sampling rate and duration of these eeg windows