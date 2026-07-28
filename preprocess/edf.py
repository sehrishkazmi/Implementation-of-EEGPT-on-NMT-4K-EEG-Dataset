# 3. Ensures there are no missing target channels in the dataset. If any are missing, it will print a warning and list them.
import pandas as pd
df = pd.read_csv(r"preprocess\eegpt_full_dataset_inspection.csv")
print("Raw Channels in File:")
print(df["all_channel_names"].iloc[0])
print("\nMissing Target Channels:")
print(df["missing_target_channels"].iloc[0])

# Confirmed: No target channels are missing with the exception of [FPZ]