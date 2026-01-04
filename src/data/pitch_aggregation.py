import pybaseball
from pybaseball import statcast
import pandas as pd
import numpy as np


def calculate_vaa_haa(df):
    """Calculate Vertical and Horizontal Attack Angles"""
    dx = df['plate_x'] - df['release_pos_x']
    dz = df['plate_z'] - df['release_pos_z']
    dy = -df['release_pos_y']  # plate at y = 0

    df['VAA'] = np.degrees(np.arctan2(dz, np.abs(dy)))
    df['HAA'] = np.degrees(np.arctan2(dx, np.abs(dy)))

    return df


def aggregate_pitch_data(start_date, end_date, save=False, output_path=None):
    """
    Aggregate pitcher statistics and descriptors from StatCast data.
    
    Parameters
    ----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    save : bool, optional
        Whether to save the output to a parquet file (default: False)
    output_path : str, optional
        Path where to save the parquet file. Required if save=True.
    
    Returns
    -------
    pd.DataFrame
        Aggregated pitch data with statistics and descriptors
    """
    
    # Fetch StatCast data
    print(f"Fetching StatCast data from {start_date} to {end_date}...")
    statcast_data = statcast(start_date, end_date)
    
    # Select columns of interest
    cols = [
        'game_date', 'pitch_type', 'p_throws', 'stand', 'events', 'description',
        'release_speed', 'release_pos_x', 'release_pos_y', 'release_pos_z',
        'player_name', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'effective_speed', 'release_spin_rate', 'release_extension',
        'pitcher', 'spin_axis'
    ]
    filt_data = statcast_data[cols]
    
    # Calculate VAA and HAA
    filt_data = calculate_vaa_haa(filt_data)
    
    # Define events of interest
    swinging_strikes = {'swinging_strike', 'swinging_strike_blocked'}
    swings = {'foul', 'foul_tip', 'swinging_strike', 'swinging_strike_blocked', 'hit_into_play'}
    called_strikes = {'called_strike'}
    balls = {'ball', 'blocked_ball'}
    
    # Prepare data for aggregation
    df = filt_data.copy()
    df['season'] = df['game_date'].dt.year
    
    # Create binary columns
    df['is_swing'] = df['description'].isin(swings)
    df['is_whiff'] = df['description'].isin(swinging_strikes)
    df['is_called_strike'] = df['description'].isin(called_strikes)
    df['is_ball'] = df['description'].isin(balls)
    
    # Aggregate predictive stats per pitcher-pitch_type-season
    print("Aggregating pitch statistics...")
    agg = (
        df.groupby(['pitcher', 'pitch_type', 'season'])
        .agg(
            pitches=('description', 'count'),
            swings=('is_swing', 'sum'),
            whiffs=('is_whiff', 'sum'),
            called_strikes=('is_called_strike', 'sum'),
            balls=('is_ball', 'sum')
        )
        .reset_index()
    )
    
    agg['whiff_pct'] = agg['whiffs'] / agg['swings']
    agg['csw_pct'] = (agg['whiffs'] + agg['called_strikes']) / agg['pitches']
    
    # Aggregate descriptive statistics
    print("Aggregating pitch descriptors...")
    desc = (
        df.groupby(['pitcher', 'pitch_type', 'season'])
        .agg(
            n_pitches=('description', 'count'),
            velo=('release_speed', 'mean'),
            pfx_x=('pfx_x', 'mean'),
            pfx_z=('pfx_z', 'mean'),
            VAA=('VAA', 'mean'),
            HAA=('HAA', 'mean'),
            ext=('release_extension', 'mean'),
            spin=('release_spin_rate', 'mean'),
            rel_x=('release_pos_x', 'mean'),
            rel_z=('release_pos_z', 'mean'),
            plate_x=('plate_x', 'mean'),
            plate_z=('plate_z', 'mean'),
        )
        .reset_index()
    )
    
    # Calculate pitch usage
    total = desc.groupby(['pitcher', 'season'])['n_pitches'].transform('sum')
    desc['usage'] = desc['n_pitches'] / total
    
    # Merge statistics with descriptors
    print("Merging data...")
    final = pd.merge(agg, desc, on=['pitcher', 'pitch_type', 'season'])
    
    # Save if requested
    if save:
        if output_path is None:
            raise ValueError("output_path must be specified when save=True")
        print(f"Saving to {output_path}...")
        final.to_parquet(output_path)
        print("Done!")
    
    return final


if __name__ == "__main__":
    # Example usage
    result = aggregate_pitch_data(
        start_date='2025-03-28',
        end_date='2025-03-29',
        save=True,
        output_path='initial_short_pitches_description_whiff_csw_stats.parquet'
    )
    print(result.head())
