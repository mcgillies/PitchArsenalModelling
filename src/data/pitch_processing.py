import numpy as np
import pandas as pd
from pathlib import Path

# All pitch types to include in flattened output
ALL_PITCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CH", "FS", "KN", "SC"]


def process_pitches(df, min_pitches=30, save=False, start_date=None, end_date=None):
    """
    Process aggregated pitch data: filter, convert movement metrics, and flatten to pitcher-level.

    Each pitcher-season row will have columns for ALL pitch types (e.g., FF_velo, CH_velo, etc.).
    Pitch types that a pitcher doesn't throw will be filled with NaN.

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
        Flattened pitcher-level data with columns for each pitch type
    """

    # Remove unnecessary columns
    filtered_pitches = df.drop(columns=['swings', 'whiffs', 'called_strikes', 'balls', 'n_pitches'], errors='ignore')

    # Convert pfx_x and pfx_z to HB and IVB in inches
    filtered_pitches['HB'] = filtered_pitches['pfx_x'] * 12
    filtered_pitches['IVB'] = filtered_pitches['pfx_z'] * 12
    filtered_pitches = filtered_pitches.drop(columns=['pfx_x', 'pfx_z'])

    # Filter on minimum number of pitches
    filtered_pitches = filtered_pitches[filtered_pitches['pitches'] >= min_pitches]

    # Calculate arsenal size per pitcher-season (before flattening)
    arsenal_size = filtered_pitches.groupby(["pitcher", "season"])["pitch_type"].nunique().reset_index()
    arsenal_size.columns = ["pitcher", "season", "arsenal_size"]

    # Get p_throws per pitcher-season (take the first value)
    p_throws = filtered_pitches.groupby(["pitcher", "season"])["p_throws"].first().reset_index()

    # Columns to pivot (all feature columns except identifiers)
    id_cols = ["pitcher", "pitch_type", "season", "p_throws"]
    feature_cols = [c for c in filtered_pitches.columns if c not in id_cols]

    # Flatten: pivot so each pitch type becomes column prefixes
    print("Flattening pitch data to pitcher-level...")

    # Create a multi-index pivot table
    pivoted = filtered_pitches.pivot_table(
        index=["pitcher", "season"],
        columns="pitch_type",
        values=feature_cols,
        aggfunc="first"
    )

    # Flatten column names: (feature, pitch_type) -> pitch_type_feature
    pivoted.columns = [f"{pitch_type}_{feature}" for feature, pitch_type in pivoted.columns]
    pivoted = pivoted.reset_index()

    # Ensure all pitch types have columns (fill missing with NaN)
    existing_pitch_types = filtered_pitches["pitch_type"].unique()
    for pitch_type in ALL_PITCH_TYPES:
        if pitch_type not in existing_pitch_types:
            # Add NaN columns for this pitch type
            for feature in feature_cols:
                col_name = f"{pitch_type}_{feature}"
                if col_name not in pivoted.columns:
                    pivoted[col_name] = np.nan

    # Get pitch_type per pitcher-season (one row per pitch type they throw)
    pitch_types_df = filtered_pitches[["pitcher", "season", "pitch_type", "pitches", 'whiff_pct']].copy()

    # Merge pivoted columns to pitch_type level (one row per pitcher-season-pitch_type)
    result = pitch_types_df.merge(pivoted, on=["pitcher", "season"], how="left")

    # Merge arsenal_size and p_throws
    result = result.merge(arsenal_size, on=["pitcher", "season"], how="left")
    result = result.merge(p_throws, on=["pitcher", "season"], how="left")

    # Reorder columns: identifiers first, then pivoted features alphabetically
    id_output_cols = ["pitcher", "season", "p_throws", "pitch_type", "arsenal_size", "pitches", "whiff_pct"]
    feature_output_cols = sorted([c for c in result.columns if c not in id_output_cols])
    result = result[id_output_cols + feature_output_cols]

    print(f"Flattened to {len(result)} pitcher-season-pitch_type rows with {len(result.columns)} columns")

    # Save if requested
    if save:
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date must be specified when save=True")
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / 'data' / f'processed_pitches_df_{start_date}_{end_date}.parquet'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {output_path}...")
        result.to_parquet(output_path)
        print("Done!")

    return result


if __name__ == "__main__":
    # Example usage
    from pitch_aggregation import aggregate_pitch_data
    
    start_date = '2025-03-30'
    end_date = '2025-04-30'
    
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
