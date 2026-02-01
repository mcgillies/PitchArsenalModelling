import pybaseball
from pybaseball import statcast
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def calculate_vaa_haa(df):
    """Calculate Vertical and Horizontal Attack Angles"""
    dx = df['plate_x'] - df['release_pos_x']
    dz = df['plate_z'] - df['release_pos_z']
    dy = -df['release_pos_y']  # plate at y = 0

    df['VAA'] = np.degrees(np.arctan2(dz, np.abs(dy)))
    df['HAA'] = np.degrees(np.arctan2(dx, np.abs(dy)))

    return df


def fetch_statcast_chunked(start_date, end_date, chunk_days=7):

    # Convert strings → datetime.date if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    print(f"Fetching StatCast data from {start_date} to {end_date} in {chunk_days}-day chunks...")

    all_chunks = []
    cur = start_date

    while cur <= end_date:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end_date)
        print(f"  → {cur} to {chunk_end}")

        try:
            df = statcast(
                cur.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d")
            )
            if not df.empty:
                all_chunks.append(df)
        except Exception as e:
            print(f"    ⚠️ Failed for {cur}–{chunk_end}: {e}")

        cur = chunk_end + timedelta(days=1)

    return pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()




# def aggregate_pitch_data(start_date, end_date, save=False, output_path=None):
#     """
#     Aggregate pitcher statistics and descriptors from StatCast data.
    
#     Parameters
#     ----------
#     start_date : str
#         Start date in format 'YYYY-MM-DD'
#     end_date : str
#         End date in format 'YYYY-MM-DD'
#     save : bool, optional
#         Whether to save the output to a parquet file (default: False)
#     output_path : str, optional
#         Path where to save the parquet file. Required if save=True.
    
#     Returns
#     -------
#     pd.DataFrame
#         Aggregated pitch data with statistics and descriptors
#     """
    
#     # Fetch StatCast data chunked. 
#     statcast_data = fetch_statcast_chunked(start_date, end_date, chunk_days=7)

    
#     # Select columns of interest
#     cols = [
#         'game_date', 'pitch_type', 'p_throws', 'stand', 'events', 'description',
#         'release_speed', 'release_pos_x', 'release_pos_y', 'release_pos_z',
#         'player_name', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
#         'effective_speed', 'release_spin_rate', 'release_extension',
#         'pitcher', 'spin_axis'
#     ]
#     filt_data = statcast_data[cols]
    
#     # Calculate VAA and HAA
#     filt_data = calculate_vaa_haa(filt_data)
    
#     # Define events of interest
#     swinging_strikes = {'swinging_strike', 'swinging_strike_blocked'}
#     swings = {'foul', 'foul_tip', 'swinging_strike', 'swinging_strike_blocked', 'hit_into_play'}
#     called_strikes = {'called_strike'}
#     balls = {'ball', 'blocked_ball'}
    
#     # Prepare data for aggregation
#     df = filt_data.copy()
#     df['season'] = df['game_date'].dt.year
    
#     # Create binary columns
#     df['is_swing'] = df['description'].isin(swings)
#     df['is_whiff'] = df['description'].isin(swinging_strikes)
#     df['is_called_strike'] = df['description'].isin(called_strikes)
#     df['is_ball'] = df['description'].isin(balls)
    
#     # Aggregate predictive stats per pitcher-pitch_type-season
#     print("Aggregating pitch statistics...")
#     agg = (
#         df.groupby(['pitcher', 'pitch_type', 'season'])
#         .agg(
#             pitches=('description', 'count'),
#             swings=('is_swing', 'sum'),
#             whiffs=('is_whiff', 'sum'),
#             called_strikes=('is_called_strike', 'sum'),
#             balls=('is_ball', 'sum')
#         )
#         .reset_index()
#     )
    
#     agg['whiff_pct'] = agg['whiffs'] / agg['swings']
#     agg['csw_pct'] = (agg['whiffs'] + agg['called_strikes']) / agg['pitches']
    
#     # Aggregate descriptive statistics
#     print("Aggregating pitch descriptors...")
#     desc = (
#         df.groupby(['pitcher', 'pitch_type', 'season'])
#         .agg(
#             n_pitches=('description', 'count'),
#             p_throws=('p_throws', 'first'),
#             velo=('release_speed', 'mean'),
#             pfx_x=('pfx_x', 'mean'),
#             pfx_z=('pfx_z', 'mean'),
#             VAA=('VAA', 'mean'),
#             HAA=('HAA', 'mean'),
#             ext=('release_extension', 'mean'),
#             spin=('release_spin_rate', 'mean'),
#             rel_x=('release_pos_x', 'mean'),
#             rel_z=('release_pos_z', 'mean'),
#             plate_x=('plate_x', 'mean'),
#             plate_z=('plate_z', 'mean'),
#         )
#         .reset_index()
#     )
    
#     # Calculate pitch usage
#     total = desc.groupby(['pitcher', 'season'])['n_pitches'].transform('sum')
#     desc['usage'] = desc['n_pitches'] / total
    
#     # Merge statistics with descriptors
#     print("Merging data...")
#     final = pd.merge(agg, desc, on=['pitcher', 'pitch_type', 'season'])
    
#     # Save if requested
#     if save:
#         if output_path is None:
#             raise ValueError("output_path must be specified when save=True")
#         output_path = Path(output_path)
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         print(f"Saving to {output_path}...")
#         final.to_parquet(output_path)
#         print("Done!")
    
#     return final

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# ZONE CONSTANTS (MLB regulation)
# Adjust these if your pitch_x/pitch_z use a different origin or units.
# ---------------------------------------------------------------------------
ZONE_LEFT   = -0.7083   # ~8.5 inches left of center, in feet
ZONE_RIGHT  =  0.7083
ZONE_BOTTOM =  1.5      # approximate low edge
ZONE_TOP    =  3.4      # approximate high edge
ZONE_CENTER_X = 0.0
ZONE_CENTER_Z = (ZONE_BOTTOM + ZONE_TOP) / 2  # ~2.45 ft


def compute_command_features(df, group_cols=["player_id", "season"], game_col="game_id"):
    """
    Computes 15 command features from pitch_x and pitch_z.

    Features
    --------
    x_mean, z_mean              - baseline location tendency
    x_std, z_std                - consistency on each axis (core command signal)
    pct_in_zone                 - most direct measure of command
    pct_on_edge                 - edge vs. middle of zone usage
    dist_mean, dist_std         - avg distance from zone center & its consistency
    x_range, z_range            - how much of the zone they work
    seq_dist_mean               - pitch-to-pitch movement (unpredictability)
    x_signed_offset             - directional bias left/right of center
    z_signed_offset             - directional bias above/below center
    xz_correlation              - whether misses tend diagonal vs. axis-aligned
    pct_high                    - vertical tendency (only need one of high/low)
    pct_outside                 - horizontal tendency (only need one of in/out)

    Parameters
    ----------
    df : pd.DataFrame
        Pitch-level data. Must have: pitch_x, pitch_z, + group_cols.
        Must have game_col if you want seq_dist_mean (set game_col=None to skip).
    group_cols : list
        Columns to group by (e.g., ["player_id", "season"]).
    game_col : str or None
        Column identifying each game. Set to None to skip sequential feature.
    """
    df = df.copy()

    # --- precompute per-pitch columns ---
    df["dist_from_center"] = np.sqrt(
        (df["pitch_x"] - ZONE_CENTER_X) ** 2 +
        (df["pitch_z"] - ZONE_CENTER_Z) ** 2
    )
    df["in_zone"] = (
        (df["pitch_x"] >= ZONE_LEFT)  & (df["pitch_x"] <= ZONE_RIGHT) &
        (df["pitch_z"] >= ZONE_BOTTOM) & (df["pitch_z"] <= ZONE_TOP)
    )

    # Edge = in zone but NOT in the inner third on both axes
    x_third = (ZONE_RIGHT - ZONE_LEFT) / 3
    z_third = (ZONE_TOP - ZONE_BOTTOM) / 3
    df["on_edge"] = df["in_zone"] & ~(
        (df["pitch_x"] >= ZONE_LEFT  + x_third) & (df["pitch_x"] <= ZONE_RIGHT - x_third) &
        (df["pitch_z"] >= ZONE_BOTTOM + z_third) & (df["pitch_z"] <= ZONE_TOP   - z_third)
    )

    df["is_high"]    = df["pitch_z"] > ZONE_CENTER_Z
    df["is_outside"] = df["pitch_x"].abs() >= (ZONE_RIGHT / 2)

    # --- sequential: pitch-to-pitch distance ---
    has_seq = game_col and game_col in df.columns
    if has_seq:
        df = df.sort_values(group_cols + [game_col])
        prev_cols = group_cols + [game_col]
        df["prev_x"] = df.groupby(prev_cols)["pitch_x"].shift(1)
        df["prev_z"] = df.groupby(prev_cols)["pitch_z"].shift(1)
        df["seq_dist"] = np.sqrt(
            (df["pitch_x"] - df["prev_x"]) ** 2 +
            (df["pitch_z"] - df["prev_z"]) ** 2
        )

    # --- aggregate ---
    aggs = {
        "x_mean":           ("pitch_x",         "mean"),
        "z_mean":           ("pitch_z",         "mean"),
        "x_std":            ("pitch_x",         "std"),
        "z_std":            ("pitch_z",         "std"),
        "x_range":          ("pitch_x",         lambda s: s.max() - s.min()),
        "z_range":          ("pitch_z",         lambda s: s.max() - s.min()),
        "pct_in_zone":      ("in_zone",         "mean"),
        "pct_on_edge":      ("on_edge",         "mean"),
        "dist_mean":        ("dist_from_center","mean"),
        "dist_std":         ("dist_from_center","std"),
        "pct_high":         ("is_high",         "mean"),
        "pct_outside":      ("is_outside",      "mean"),
    }

    if has_seq:
        aggs["seq_dist_mean"] = ("seq_dist", "mean")

    features = df.groupby(group_cols).agg(**aggs).reset_index()

    # --- signed offset & correlation need a manual loop (can't do in .agg) ---
    offset_corr = []
    for name, group in df.groupby(group_cols):
        row = dict(zip(group_cols, name)) if isinstance(name, tuple) else {group_cols[0]: name}
        row["x_signed_offset"] = (group["pitch_x"] - ZONE_CENTER_X).mean()
        row["z_signed_offset"] = (group["pitch_z"] - ZONE_CENTER_Z).mean()
        row["xz_correlation"]  = group["pitch_x"].corr(group["pitch_z"]) if len(group) > 2 else np.nan
        offset_corr.append(row)

    features = features.merge(pd.DataFrame(offset_corr), on=group_cols, how="left")

    print(f"Done. {features.shape[1]} features, {features.shape[0]} rows.")
    return features


def aggregate_pitch_data(start_date, end_date, save=False, output_path=None):
    """
    Aggregate pitcher statistics and descriptors from StatCast data, including
    robust (coarse) location-intent features derived from plate_x/plate_z.

    Returns one row per (pitcher, pitch_type, season) with:
      - targets: whiff_pct, csw_pct (+ counts)
      - descriptors: velo, pfx_x, pfx_z, VAA, HAA, ext, spin, rel_x, rel_z, usage
      - coarse location intent features:
          above_zone_rate, below_zone_rate,
          zbin_low_rate, zbin_mid_rate, zbin_high_rate,
          glove_side_rate, arm_side_rate,
          mean_z_dist_mid, mean_x_dist_center,
          edge_rate,
          pct_vs_RHB

    Raw per-pitch columns (plate_x, plate_z, etc.) are NOT kept in the output
    (only aggregated features are returned).
    """

    # --- 0) Pull ---
    statcast_data = fetch_statcast_chunked(start_date, end_date, chunk_days=7)

    # --- 1) Select columns ---
    cols = [
        "game_date", "pitch_type", "p_throws", "stand", "events", "description",
        "release_speed", "release_pos_x", "release_pos_y", "release_pos_z",
        "pfx_x", "pfx_z", "plate_x", "plate_z",
        "release_spin_rate", "release_extension",
        "pitcher",
    ]
    # only keep cols that exist (some seasons omit certain fields)
    cols = [c for c in cols if c in statcast_data.columns]
    df = statcast_data[cols].copy()

    # Ensure datetime
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["season"] = df["game_date"].dt.year

    # --- 2) Calculate VAA/HAA (your existing function should use release_pos_* and plate_*) ---
    df = calculate_vaa_haa(df)

    # --- 3) Outcomes flags ---
    swinging_strikes = {"swinging_strike", "swinging_strike_blocked"}
    swings = {"foul", "foul_tip", "swinging_strike", "swinging_strike_blocked", "hit_into_play"}
    called_strikes = {"called_strike"}
    balls = {"ball", "blocked_ball"}

    df["is_swing"] = df["description"].isin(swings)
    df["is_whiff"] = df["description"].isin(swinging_strikes)
    df["is_called_strike"] = df["description"].isin(called_strikes)
    df["is_ball"] = df["description"].isin(balls)

    # --- 4) Coarse location intent features (pitch-level flags) ---
    # Defensive numeric coercion
    df["plate_x"] = pd.to_numeric(df["plate_x"], errors="coerce")
    df["plate_z"] = pd.to_numeric(df["plate_z"], errors="coerce")

    # Batter mix (optional but useful for context)
    df["is_vs_RHB"] = (df["stand"] == "R").astype(int)

    # Simple zone thresholds (ft). These are intentionally coarse and stable.
    # Z_LOW = 1.8
    # Z_HIGH = 3.5
    # Z_MID = 2.65
    # X_EDGE = 0.7

    # df["above_zone"] = df["plate_z"] > Z_HIGH
    # df["below_zone"] = df["plate_z"] < Z_LOW

    # # z bins: low / mid / high
    # df["zbin_low"] = df["plate_z"] < Z_LOW
    # df["zbin_high"] = df["plate_z"] > Z_HIGH
    # df["zbin_mid"] = (~df["zbin_low"]) & (~df["zbin_high"]) & df["plate_z"].notna()

    # # handedness-adjusted horizontal location so "glove-side" is consistent:
    # # for LHB, flip sign so + means glove-side (relative to pitcher)
    # df["plate_x_adj"] = np.where(df["stand"] == "L", -df["plate_x"], df["plate_x"])
    # df["x_glove_side"] = df["plate_x_adj"] > X_EDGE
    # df["x_arm_side"] = df["plate_x_adj"] < -X_EDGE

    # # Distance-based (stable)
    # df["z_dist_mid"] = (df["plate_z"] - Z_MID).abs()
    # df["x_dist_center"] = df["plate_x"].abs()

    # # Simple "edge" definition: near sides while in the vertical middle band
    # df["edge"] = (df["plate_x"].abs() > X_EDGE) & df["plate_z"].between(Z_LOW, Z_HIGH)

    # --- 5) Aggregate targets per pitcher-pitch_type-season ---
    print("Aggregating pitch statistics...")
    agg = (
        df.groupby(["pitcher", "pitch_type", "season"], dropna=False)
          .agg(
              pitches=("description", "count"),
              swings=("is_swing", "sum"),
              whiffs=("is_whiff", "sum"),
              called_strikes=("is_called_strike", "sum"),
              balls=("is_ball", "sum"),
          )
          .reset_index()
    )

    # Safe divisions
    agg["whiff_pct"] = np.where(agg["swings"] > 0, agg["whiffs"] / agg["swings"], np.nan)
    agg["csw_pct"] = (agg["whiffs"] + agg["called_strikes"]) / agg["pitches"]

    # --- 6) Aggregate descriptors + intent features ---
    print("Aggregating pitch descriptors + intent features...")
    desc = (
        df.groupby(["pitcher", "pitch_type", "season"], dropna=False)
          .agg(
              n_pitches=("description", "count"),
              p_throws=("p_throws", "first"),

              # core descriptors
              velo=("release_speed", "mean"),
              pfx_x=("pfx_x", "mean"),
              pfx_z=("pfx_z", "mean"),
              VAA=("VAA", "mean"),
              HAA=("HAA", "mean"),
              ext=("release_extension", "mean"),
              spin=("release_spin_rate", "mean"),
              rel_x=("release_pos_x", "mean"),
              rel_z=("release_pos_z", "mean"),

              # handedness mix
              pct_vs_RHB=("is_vs_RHB", "mean"),

              # intent features (rates)
              above_zone_rate=("above_zone", "mean"),
              below_zone_rate=("below_zone", "mean"),
              zbin_low_rate=("zbin_low", "mean"),
              zbin_mid_rate=("zbin_mid", "mean"),
              zbin_high_rate=("zbin_high", "mean"),
              glove_side_rate=("x_glove_side", "mean"),
              arm_side_rate=("x_arm_side", "mean"),
              edge_rate=("edge", "mean"),

              # intent features (distances)
              mean_z_dist_mid=("z_dist_mid", "mean"),
              mean_x_dist_center=("x_dist_center", "mean"),
          )
          .reset_index()
    )

    # Usage
    total = desc.groupby(["pitcher", "season"])["n_pitches"].transform("sum")
    desc["usage"] = desc["n_pitches"] / total

    # --- 7) Merge ---
    print("Merging data...")
    final = pd.merge(agg, desc, on=["pitcher", "pitch_type", "season"], how="inner")

    # --- 8) Drop raw columns if any slipped in (defensive) ---
    raw_like = {"plate_x", "plate_z", "plate_x_adj", "game_date", "description", "events", "stand"}
    final = final.drop(columns=[c for c in final.columns if c in raw_like], errors="ignore")

    # --- 9) Save ---
    if save:
        if output_path is None:
            raise ValueError("output_path must be specified when save=True")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {output_path}...")
        final.to_parquet(output_path, index=False)
        print("Done!")

    return final



if __name__ == "__main__":
    # Example usage
    start_date = '2025-03-28'
    end_date = '2025-03-29'
    
    # Resolve to project root data folder
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / 'data' / f'{start_date}_{end_date}_pitches_description_whiff_csw_stats.parquet'
    
    result = aggregate_pitch_data(
        start_date=start_date,
        end_date=end_date,
        save=True,
        output_path=str(output_path)
    )
    print(result.head())
