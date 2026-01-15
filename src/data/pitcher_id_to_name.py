import pandas as pd
from pybaseball import playerid_reverse_lookup

# Set working dir to root:
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("CWD:", os.getcwd())

# Move back a directory to repo root
os.chdir("..")


# Load your processed file
df = pd.read_parquet("data/processed_pitches_df_2023-03-30_2025-09-30.parquet")

ids = sorted(df["pitcher"].dropna().astype(int).unique().tolist())

# Reverse lookup (MLBAM -> names)
lu = playerid_reverse_lookup(ids, key_type="mlbam")

# Create a clean label
lu["player_name"] = lu["name_first"].fillna("") + " " + lu["name_last"].fillna("")
lu["player_name"] = lu["player_name"].str.strip()

# Keep only what you need
pitcher_map = lu[["key_mlbam", "player_name"]].rename(columns={"key_mlbam": "pitcher"})

# Save mapping for the app to reuse
pitcher_map.to_parquet("data/pitcher_id_to_name.parquet", index=False)

print("Saved:", pitcher_map.shape)
pitcher_map.head()
