import numpy as np
import pandas as pd
from pathlib import Path

FASTBALL_TYPES_DEFAULT = ("FF", "SI", "FC", "FT")


def add_arsenal_context_features(
    df: pd.DataFrame,
    fastball_types=FASTBALL_TYPES_DEFAULT,
    fb_min_usage: float = 0.0,
) -> pd.DataFrame:
    """
    Adds arsenal context features to a season-level pitch table.

    Expected input columns (your filtered_pitches):
      ['pitcher','pitch_type','season','usage','velo','HB','IVB','VAA','HAA','ext','spin','rel_x','rel_z', ...]
    Output:
      - fb_* columns (primary fastball descriptors per pitcher-season)
      - delta_* columns (pitch minus fb)
      - arsenal summaries: arsenal_size, second_pitch_usage, max_other_usage
      - redundancy: min_movdist_to_other, min_shape_dist_to_other
    """

    req = {"pitcher", "pitch_type", "season", "usage", "velo", "HB", "IVB", "VAA", "HAA", "ext", "spin", "rel_x", "rel_z"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    key_cols = ["pitcher", "season"]

    # --- 1) Identify primary fastball per pitcher-season (highest usage among fastball types) ---
    fb = out[out["pitch_type"].isin(fastball_types)].copy()
    if fb_min_usage > 0:
        fb = fb[fb["usage"] >= fb_min_usage]

    fb_idx = fb.groupby(key_cols)["usage"].idxmax()
    fb_primary = fb.loc[fb_idx, key_cols + ["pitch_type", "usage", "velo", "HB", "IVB", "VAA", "HAA", "ext", "spin", "rel_x", "rel_z"]].copy()

    fb_primary = fb_primary.rename(columns={
        "pitch_type": "fb_pitch_type",
        "usage": "fb_usage",
        "velo": "fb_velo",
        "HB": "fb_HB",
        "IVB": "fb_IVB",
        "VAA": "fb_VAA",
        "HAA": "fb_HAA",
        "ext": "fb_ext",
        "spin": "fb_spin",
        "rel_x": "fb_rel_x",
        "rel_z": "fb_rel_z",
    })

    out = out.merge(fb_primary, on=key_cols, how="left")

    # --- 2) Delta features vs primary fastball ---
    for base, fbcol in [
        ("velo", "fb_velo"),
        ("HB", "fb_HB"),
        ("IVB", "fb_IVB"),
        ("VAA", "fb_VAA"),
        ("HAA", "fb_HAA"),
        ("ext", "fb_ext"),
        ("spin", "fb_spin"),
        ("rel_x", "fb_rel_x"),
        ("rel_z", "fb_rel_z"),
    ]:
        out[f"delta_{base}_vs_fb"] = out[base] - out[fbcol]

    out["movdist_vs_fb"] = np.sqrt(
        (out["HB"] - out["fb_HB"]) ** 2 + (out["IVB"] - out["fb_IVB"]) ** 2
    )

    velo_scale = 2.0
    out["shape_dist_vs_fb"] = np.sqrt(
        (velo_scale * (out["velo"] - out["fb_velo"])) ** 2
        + (out["HB"] - out["fb_HB"]) ** 2
        + (out["IVB"] - out["fb_IVB"]) ** 2
    )

    # --- 3) Arsenal summary features (per pitcher-season) ---
    # Changed to calculate before filtering
    # out["arsenal_size"] = out.groupby(key_cols)["pitch_type"].transform("nunique")

    def second_highest(s: pd.Series) -> float:
        vals = np.sort(s.to_numpy())
        return float(vals[-2]) if len(vals) >= 2 else np.nan

    out["second_pitch_usage"] = out.groupby(key_cols)["usage"].transform(second_highest)

    def top_two(s: pd.Series):
        v = np.sort(s.to_numpy())
        if len(v) == 1:
            return (v[-1], np.nan)
        return (v[-1], v[-2])

    top2 = out.groupby(key_cols)["usage"].apply(top_two).reset_index(name="top2")
    top2[["g_max_usage", "g_second_usage"]] = pd.DataFrame(top2["top2"].tolist(), index=top2.index)
    top2 = top2.drop(columns=["top2"])

    out = out.merge(top2, on=key_cols, how="left")

    out["max_other_usage"] = np.where(
        np.isclose(out["usage"], out["g_max_usage"]),
        out["g_second_usage"],
        out["g_max_usage"]
    )
    out = out.drop(columns=["g_max_usage", "g_second_usage"])

    # --- 4) Redundancy features: min distance to any OTHER pitch in the arsenal ---
    def add_min_distances(group: pd.DataFrame) -> pd.DataFrame:
        hb = group["HB"].to_numpy()
        ivb = group["IVB"].to_numpy()
        velo = group["velo"].to_numpy()

        n = len(group)
        if n <= 1:
            group["min_movdist_to_other"] = np.nan
            group["min_shape_dist_to_other"] = np.nan
            return group

        d_mov = np.sqrt((hb[:, None] - hb[None, :]) ** 2 + (ivb[:, None] - ivb[None, :]) ** 2)

        d_shape = np.sqrt((velo_scale * (velo[:, None] - velo[None, :])) ** 2 +
                          (hb[:, None] - hb[None, :]) ** 2 +
                          (ivb[:, None] - ivb[None, :]) ** 2)

        np.fill_diagonal(d_mov, np.inf)
        np.fill_diagonal(d_shape, np.inf)

        group["min_movdist_to_other"] = d_mov.min(axis=1)
        group["min_shape_dist_to_other"] = d_shape.min(axis=1)
        return group

    out = out.groupby(key_cols, group_keys=False).apply(add_min_distances)

    return out


def process_pitches(df, min_pitches=30, save=False, start_date=None, end_date=None):
    """
    Process aggregated pitch data: filter, convert movement metrics, and add arsenal features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Output from aggregate_pitch_data()
    min_pitches : int, optional
        Minimum number of pitches per pitch type to retain (default: 30)
    save : bool, optional
        Whether to save the output to a parquet file (default: False)
    start_date : str, optional
        Start date for the parquet filename. Required if save=True.
    end_date : str, optional
        End date for the parquet filename. Required if save=True.
    
    Returns
    -------
    pd.DataFrame
        Processed pitch data with arsenal context features
    """
    
    # Remove unnecessary columns
    filtered_pitches = df.drop(columns=['swings', 'whiffs', 'called_strikes', 'balls', 'n_pitches'])
    
    # Convert pfx_x and pfx_z to HB and IVB in inches
    filtered_pitches['HB'] = filtered_pitches['pfx_x'] * 12
    filtered_pitches['IVB'] = filtered_pitches['pfx_z'] * 12
    filtered_pitches = filtered_pitches.drop(columns=['pfx_x', 'pfx_z'])
    
    # Filter on minimum number of pitches
    filtered_pitches = filtered_pitches[filtered_pitches['pitches'] >= min_pitches]

    # Create arsenal size column before filtering
    filtered_pitches["arsenal_size"] = filtered_pitches.groupby(["pitcher", "season"])["pitch_type"].transform("nunique")
    
    # Add arsenal context features
    print("Adding arsenal context features...")
    arsenal_df = add_arsenal_context_features(filtered_pitches)
    
    # Save if requested
    if save:
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date must be specified when save=True")
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / 'data' / f'processed_pitches_df_{start_date}_{end_date}.parquet'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {output_path}...")
        arsenal_df.to_parquet(output_path)
        print("Done!")
    
    return arsenal_df


if __name__ == "__main__":
    # Example usage
    from pitch_aggregation import aggregate_pitch_data
    
    start_date = '2023-03-30'
    end_date = '2025-09-30'
    
    project_root = Path(__file__).parent.parent.parent
    agg_output_path = project_root / 'data' / f'{start_date}_{end_date}_pitches_description_whiff_csw_stats.parquet'
    
    # Fetch and aggregate pitch data
    agg_df = aggregate_pitch_data(
        start_date=start_date,
        end_date=end_date,
        save=True,
        output_path=str(agg_output_path)
    )

    # Process with arsenal features
    processed_df = process_pitches(
        agg_df,
        min_pitches=30,
        save=True,
        start_date=start_date,
        end_date=end_date
    )
    
    print(processed_df.head())
