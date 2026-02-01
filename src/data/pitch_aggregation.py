import pybaseball
from pybaseball import statcast
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# ZONE CONSTANTS (MLB regulation)
# ---------------------------------------------------------------------------
ZONE_LEFT = -0.7083      # ~8.5 inches left of center, in feet
ZONE_RIGHT = 0.7083
ZONE_BOTTOM = 1.5        # approximate low edge
ZONE_TOP = 3.4           # approximate high edge
ZONE_CENTER_X = 0.0
ZONE_CENTER_Z = (ZONE_BOTTOM + ZONE_TOP) / 2  # ~2.45 ft

# Coarse zone thresholds
Z_LOW = 1.8
Z_HIGH = 3.5
Z_MID = 2.65
X_EDGE = 0.7


def calculate_vaa_haa(df):
    """Calculate Vertical and Horizontal Attack Angles"""
    dx = df['plate_x'] - df['release_pos_x']
    dz = df['plate_z'] - df['release_pos_z']
    dy = -df['release_pos_y']  # plate at y = 0

    df['VAA'] = np.degrees(np.arctan2(dz, np.abs(dy)))
    df['HAA'] = np.degrees(np.arctan2(dx, np.abs(dy)))

    return df


def fetch_statcast_chunked(start_date, end_date, chunk_days=7):
    """Fetch StatCast data in chunks to avoid timeout issues."""
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


def _compute_pitch_level_features(df):
    """
    Compute all pitch-level features before aggregation.

    This includes:
    - Outcome flags (swing, whiff, called strike, ball)
    - Location intent features (zone rates, edge rates, etc.)
    - Command features (distance from center, in zone, on edge, etc.)
    """
    # Ensure numeric plate coordinates
    df["plate_x"] = pd.to_numeric(df["plate_x"], errors="coerce")
    df["plate_z"] = pd.to_numeric(df["plate_z"], errors="coerce")

    # --- Outcome flags ---
    swinging_strikes = {"swinging_strike", "swinging_strike_blocked"}
    swings = {"foul", "foul_tip", "swinging_strike", "swinging_strike_blocked", "hit_into_play"}
    called_strikes = {"called_strike"}
    balls = {"ball", "blocked_ball"}

    df["is_swing"] = df["description"].isin(swings)
    df["is_whiff"] = df["description"].isin(swinging_strikes)
    df["is_called_strike"] = df["description"].isin(called_strikes)
    df["is_ball"] = df["description"].isin(balls)

    # --- Batter handedness ---
    df["is_vs_RHB"] = (df["stand"] == "R").astype(int)

    # --- Coarse location intent features ---
    df["above_zone"] = df["plate_z"] > Z_HIGH
    df["below_zone"] = df["plate_z"] < Z_LOW

    # z bins: low / mid / high
    df["zbin_low"] = df["plate_z"] < Z_LOW
    df["zbin_high"] = df["plate_z"] > Z_HIGH
    df["zbin_mid"] = (~df["zbin_low"]) & (~df["zbin_high"]) & df["plate_z"].notna()

    # Handedness-adjusted horizontal location (glove-side is consistent)
    df["plate_x_adj"] = np.where(df["stand"] == "L", -df["plate_x"], df["plate_x"])
    df["x_glove_side"] = df["plate_x_adj"] > X_EDGE
    df["x_arm_side"] = df["plate_x_adj"] < -X_EDGE

    # Distance-based features
    df["z_dist_mid"] = (df["plate_z"] - Z_MID).abs()
    df["x_dist_center"] = df["plate_x"].abs()

    # Edge definition: near sides while in vertical middle band
    df["edge"] = (df["plate_x"].abs() > X_EDGE) & df["plate_z"].between(Z_LOW, Z_HIGH)

    # --- Command features (from compute_command_features) ---
    df["dist_from_center"] = np.sqrt(
        (df["plate_x"] - ZONE_CENTER_X) ** 2 +
        (df["plate_z"] - ZONE_CENTER_Z) ** 2
    )

    df["in_zone"] = (
        (df["plate_x"] >= ZONE_LEFT) & (df["plate_x"] <= ZONE_RIGHT) &
        (df["plate_z"] >= ZONE_BOTTOM) & (df["plate_z"] <= ZONE_TOP)
    )

    # Edge = in zone but NOT in the inner third on both axes
    x_third = (ZONE_RIGHT - ZONE_LEFT) / 3
    z_third = (ZONE_TOP - ZONE_BOTTOM) / 3
    df["on_edge"] = df["in_zone"] & ~(
        (df["plate_x"] >= ZONE_LEFT + x_third) & (df["plate_x"] <= ZONE_RIGHT - x_third) &
        (df["plate_z"] >= ZONE_BOTTOM + z_third) & (df["plate_z"] <= ZONE_TOP - z_third)
    )

    df["is_high"] = df["plate_z"] > ZONE_CENTER_Z
    df["is_outside"] = df["plate_x"].abs() >= (ZONE_RIGHT / 2)

    return df


def _compute_sequential_features(df, group_cols, game_col="game_pk"):
    """Compute pitch-to-pitch sequential distance within games."""
    if game_col not in df.columns:
        return df

    df = df.sort_values(group_cols + [game_col])
    prev_cols = group_cols + [game_col]
    df["prev_x"] = df.groupby(prev_cols)["plate_x"].shift(1)
    df["prev_z"] = df.groupby(prev_cols)["plate_z"].shift(1)
    df["seq_dist"] = np.sqrt(
        (df["plate_x"] - df["prev_x"]) ** 2 +
        (df["plate_z"] - df["prev_z"]) ** 2
    )
    return df


def _compute_correlation_features(df, group_cols):
    """Compute signed offset and correlation features that require manual iteration."""
    offset_corr = []
    for name, group in df.groupby(group_cols, dropna=False):
        if isinstance(name, tuple):
            row = dict(zip(group_cols, name))
        else:
            row = {group_cols[0]: name}

        row["x_signed_offset"] = (group["plate_x"] - ZONE_CENTER_X).mean()
        row["z_signed_offset"] = (group["plate_z"] - ZONE_CENTER_Z).mean()
        try:
            row["xz_correlation"] = group["plate_x"].corr(group["plate_z"]) if len(group) > 2 else np.nan
        except (AttributeError, ValueError):
            row["xz_correlation"] = np.nan
        offset_corr.append(row)

    return pd.DataFrame(offset_corr)


def aggregate_pitch_data(start_date, end_date, group_by_pitch_type=True, save=False, output_path=None):
    """
    Aggregate pitcher statistics, descriptors, and command features from StatCast data.

    This function fetches raw pitch data and computes all features in a single pass:
    - Targets: whiff_pct, csw_pct (+ counts)
    - Pitch descriptors: velo, movement, VAA, HAA, extension, spin, release point
    - Location intent features: zone rates, edge rates, vertical/horizontal tendencies
    - Command features: location consistency (std), zone control, spread metrics

    Parameters
    ----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    group_by_pitch_type : bool, default True
        If True, aggregate by (pitcher, pitch_type, season).
        If False, aggregate by (pitcher, season) only.
    save : bool, default False
        Whether to save the output to a parquet file
    output_path : str, optional
        Path where to save the parquet file. Required if save=True.

    Returns
    -------
    pd.DataFrame
        Aggregated pitch data with all features
    """
    # --- 1) Fetch data ---
    statcast_data = fetch_statcast_chunked(start_date, end_date, chunk_days=7)

    if statcast_data.empty:
        print("No data fetched.")
        return pd.DataFrame()

    # --- 2) Select columns ---
    cols = [
        "game_date", "game_pk", "pitch_type", "p_throws", "stand", "events", "description",
        "release_speed", "release_pos_x", "release_pos_y", "release_pos_z",
        "pfx_x", "pfx_z", "plate_x", "plate_z",
        "release_spin_rate", "release_extension",
        "pitcher",
    ]
    cols = [c for c in cols if c in statcast_data.columns]
    df = statcast_data[cols].copy()

    # Ensure datetime and extract season
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["season"] = df["game_date"].dt.year

    # --- 3) Calculate VAA/HAA ---
    df = calculate_vaa_haa(df)

    # --- 4) Compute all pitch-level features ---
    print("Computing pitch-level features...")
    df = _compute_pitch_level_features(df)

    # Define grouping columns
    if group_by_pitch_type:
        group_cols = ["pitcher", "pitch_type", "season"]
    else:
        group_cols = ["pitcher", "season"]

    # --- 5) Compute sequential features ---
    if "game_pk" in df.columns:
        df = _compute_sequential_features(df, group_cols, game_col="game_pk")
        has_seq = True
    else:
        has_seq = False

    # --- 6) Build aggregation dictionary ---
    print("Aggregating all features...")

    agg_dict = {
        # Counts and outcomes
        "pitches": ("description", "count"),
        "swings": ("is_swing", "sum"),
        "whiffs": ("is_whiff", "sum"),
        "called_strikes": ("is_called_strike", "sum"),
        "balls": ("is_ball", "sum"),

        # Pitch descriptors
        "p_throws": ("p_throws", "first"),
        "velo": ("release_speed", "mean"),
        "pfx_x": ("pfx_x", "mean"),
        "pfx_z": ("pfx_z", "mean"),
        "VAA": ("VAA", "mean"),
        "HAA": ("HAA", "mean"),
        "ext": ("release_extension", "mean"),
        "spin": ("release_spin_rate", "mean"),
        "rel_x": ("release_pos_x", "mean"),
        "rel_z": ("release_pos_z", "mean"),

        # Handedness mix
        "pct_vs_RHB": ("is_vs_RHB", "mean"),

        # Location features (limited set)
        "mean_x": ("plate_x", "mean"),
        "mean_z": ("plate_z", "mean"),
        "std_x": ("plate_x", "std"),
        "std_z": ("plate_z", "std"),
        "pct_in_zone": ("in_zone", "mean"),
    }

    # Perform aggregation
    result = df.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()

    # --- 8) Compute derived metrics ---
    # Safe divisions for rates
    result["whiff_pct"] = np.where(result["swings"] > 0, result["whiffs"] / result["swings"], np.nan)
    result["csw_pct"] = (result["whiffs"] + result["called_strikes"]) / result["pitches"]

    # Usage within pitcher-season
    if group_by_pitch_type:
        total = result.groupby(["pitcher", "season"])["pitches"].transform("sum")
        result["usage"] = result["pitches"] / total

    # --- 9) Reorder columns for clarity ---
    # Put key identifiers and targets first
    priority_cols = group_cols + ["pitches", "whiff_pct", "csw_pct"]
    if group_by_pitch_type:
        priority_cols.append("usage")

    other_cols = [c for c in result.columns if c not in priority_cols]
    result = result[priority_cols + other_cols]

    print(f"Done. {result.shape[1]} columns, {result.shape[0]} rows.")

    # --- 10) Save if requested ---
    if save:
        if output_path is None:
            raise ValueError("output_path must be specified when save=True")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {output_path}...")
        result.to_parquet(output_path, index=False)
        print("Saved!")

    return result


if __name__ == "__main__":
    # Example usage
    start_date = '2025-03-28'
    end_date = '2025-03-29'

    # Resolve to project root data folder
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / 'data' / f'{start_date}_{end_date}_pitches_aggregated.parquet'

    result = aggregate_pitch_data(
        start_date=start_date,
        end_date=end_date,
        group_by_pitch_type=True,
        save=True,
        output_path=str(output_path)
    )
    print(result.head())
