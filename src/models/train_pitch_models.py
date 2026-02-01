import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
import json


def get_next_version(models_dir):
    """
    Find the next available version number in models directory.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    existing_versions = [d.name for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
    if not existing_versions:
        return 1
    
    version_numbers = [int(v[1:]) for v in existing_versions if v[1:].isdigit()]
    return max(version_numbers) + 1 if version_numbers else 1


def preprocess_pitch_data(df, pitch_type, cat_cols):
    """
    Filter data for a specific pitch type and prepare train/val split.
    
    Returns
    -------
    tuple
        (X_train, X_val, y_train, y_val, train_df, val_df)
    """
    pitch_df = df[df["pitch_type"] == pitch_type].copy()
    
    if pitch_df.empty:
        raise ValueError(f"No data for pitch type {pitch_type}")
    
    # Train/val split by pitcher
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(pitch_df, groups=pitch_df["pitcher"]))
    
    train = pitch_df.iloc[train_idx].copy()
    val = pitch_df.iloc[val_idx].copy()
    
    # Drop non-feature columns
    drop_cols = ["pitcher", "season", "pitch_type", "whiff_pct", "csw_pct"]
    X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
    X_val = val.drop(columns=[c for c in drop_cols if c in val.columns])
    
    y_train = train["whiff_pct"].values
    y_val = val["whiff_pct"].values
    
    # One-hot encode categorical columns
    X_train = pd.get_dummies(X_train, columns=cat_cols, dummy_na=True)
    X_val = pd.get_dummies(X_val, columns=cat_cols, dummy_na=True)
    
    # Align columns
    X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)
    
    return X_train, X_val, y_train, y_val, train, val


def train_pitch_model(X_train, y_train, w_train, random_state=42):
    """
    Train an XGBoost model for a pitch type.
    """
    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=5,
        objective="reg:squarederror",
        random_state=random_state
    )
    
    model.fit(X_train, y_train, sample_weight=w_train)
    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluate model performance and return metrics.
    """
    pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)
    
    return {
        "rmse": float(rmse),
        "r2": float(r2),
        "predictions": pred.tolist()
    }


def train_all_pitch_models(
    data_path,
    models_dir="models",
    min_pitches=60,
    pitch_types=None,
    cat_cols=None
):
    """
    Train and save XGBoost models for each pitch type.
    
    Parameters
    ----------
    data_path : str or Path
        Path to the processed pitch data parquet file
    models_dir : str or Path
        Directory to save models (default: 'models')
    min_pitches : int
        Minimum number of pitches for a pitch type to be trained (default: 60)
    pitch_types : list, optional
        List of pitch types to train. If None, trains all available.
    cat_cols : list, optional
        Categorical columns for one-hot encoding (default: ['p_throws', 'fb_pitch_type'])
    
    Returns
    -------
    dict
        Training results with metrics for each pitch type
    """
    
    if cat_cols is None:
        cat_cols = ['p_throws', 'fb_pitch_type']
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Drop NAs
    df = df.dropna()
    
    # Get pitch types to train
    if pitch_types is None:
        # Get all pitch types with sufficient data
        pitch_type_counts = df.groupby("pitch_type").size()
        pitch_types = pitch_type_counts[pitch_type_counts >= min_pitches].index.tolist()
    
    print(f"Training models for pitch types: {pitch_types}")
    
    # Get version number
    version = get_next_version(models_dir)
    version_dir = Path(models_dir) / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "version": version,
        "models_dir": str(version_dir),
        "pitch_models": {}
    }
    
    # Train model for each pitch type
    for pitch_type in pitch_types:
        print(f"\n--- Training {pitch_type} ---")
        
        try:
            # Preprocess data
            X_train, X_val, y_train, y_val, train_df, val_df = preprocess_pitch_data(
                df, pitch_type, cat_cols
            )
            
            # Get sample weights
            w_train = train_df["pitches"].values
            
            # Train model
            print(f"  Training on {len(X_train)} samples...")
            model = train_pitch_model(X_train, y_train, w_train)
            
            # Evaluate
            metrics = evaluate_model(model, X_val, y_val)
            print(f"  RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            
            # Save model
            model_path = version_dir / f"{pitch_type}_model.json"
            model.get_booster().save_model(str(model_path))
            print(f"  Saved to {model_path}")
            
            # Save metadata
            results["pitch_models"][pitch_type] = {
                "model_path": str(model_path),
                "metrics": metrics,
                "feature_names": X_train.columns.tolist(),
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val)
            }
        
        except Exception as e:
            print(f"  ✗ Error training {pitch_type}: {e}")
            results["pitch_models"][pitch_type] = {
                "error": str(e)
            }
    
    # Save results summary
    results_path = version_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return results


if __name__ == "__main__":
    from pathlib import Path
    
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "processed_pitches_df_2023-03-30_2025-09-30.parquet"
    models_dir = project_root / "models"
    
    # Train all models
    results = train_all_pitch_models(
        data_path=data_path,
        models_dir=models_dir,
        min_pitches=60
    )
    
    print("\n" + "="*50)
    print(f"Training complete! Models saved in v{results['version']}/")
    for pitch_type, info in results["pitch_models"].items():
        if "error" not in info:
            print(f"  {pitch_type}: R² = {info['metrics']['r2']:.4f}")
