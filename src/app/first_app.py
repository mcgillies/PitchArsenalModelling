# src/app/first_app.py
# Run: streamlit run src/app/first_app.py
#
# Requirements:
#   pip install streamlit plotly streamlit-plotly-events pybaseball

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from pybaseball import playerid_reverse_lookup

try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False


st.set_page_config(layout="wide")
st.title("Pitch Arsenal Builder")

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[2]  # repo root if file is src/app/first_app.py
DATA_PATH = ROOT / "data" / "processed_pitches_df_2023-03-30_2025-09-30.parquet"
PITCHER_MAP_PATH = ROOT / "data" / "pitcher_id_to_name.parquet"

# Base columns needed for the app (wide format: pitch characteristics are prefixed)
BASE_COLS = ["pitcher", "season", "pitch_type"]

# Feature suffixes that exist for each pitch type (e.g., FF_velo, CH_HB)
FEATURE_SUFFIXES = [
    "usage", "velo", "HB", "IVB", "VAA", "HAA", "spin", "ext", "rel_x", "rel_z",
    "mean_x", "mean_z", "std_x", "std_z", "pct_in_zone", "pct_vs_RHB"
]

# All pitch types that may have prefixed columns in the data
PITCH_TYPES = ["FF", "SI", "FC", "FT", "SL", "CU", "CH", "FS", "KC", "ST", "SV", "CS", "FA", "FO", "EP", "KN", "SC"]

# Strike zone constants (in feet, catcher's view)
ZONE_LEFT = -0.7083      # ~8.5 inches left of center
ZONE_RIGHT = 0.7083      # ~8.5 inches right of center
ZONE_BOTTOM = 1.5        # approximate low edge
ZONE_TOP = 3.5           # approximate high edge

# Home plate dimensions (in feet)
PLATE_WIDTH = 17 / 12    # 17 inches in feet


# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_processed(path: str) -> pd.DataFrame:
    """Load the wide-format parquet where pitch features are prefixed (e.g., FF_velo, CH_HB)."""
    df = pd.read_parquet(path)

    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}")

    # Convert prefixed feature columns to numeric
    for pt in PITCH_TYPES:
        for feat in FEATURE_SUFFIXES:
            col = f"{pt}_{feat}"
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce").astype("Int64")
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["pitch_type"] = df["pitch_type"].astype("string")

    return df


def extract_arsenal_from_wide(df: pd.DataFrame, pitcher_id: int, season: int) -> pd.DataFrame:
    """
    Extract an arsenal DataFrame from wide-format data.

    The source data has one row per pitch_type with all pitch characteristics as prefixed columns.
    This function converts it to a long-format arsenal with one row per pitch type.
    """
    # Get one row for this pitcher-season (all rows have the same prefixed values)
    sub = df[(df["pitcher"] == pitcher_id) & (df["season"] == season)]
    if len(sub) == 0:
        return pd.DataFrame(columns=["pitch_type"] + FEATURE_SUFFIXES)

    row = sub.iloc[0]  # All rows have same values for prefixed columns

    # Get pitcher handedness
    p_throws = row.get("p_throws", "R")

    # Find which pitch types this pitcher actually throws (non-null usage)
    arsenal_rows = []
    for pt in PITCH_TYPES:
        usage_col = f"{pt}_usage"
        if usage_col in df.columns and pd.notna(row.get(usage_col)) and row.get(usage_col, 0) > 0:
            pitch_data = {"pitch_type": pt}
            for feat in FEATURE_SUFFIXES:
                col = f"{pt}_{feat}"
                if col in df.columns:
                    pitch_data[feat] = row.get(col)
                else:
                    # Defaults for missing features
                    if feat in ["mean_x", "mean_z"]:
                        pitch_data[feat] = 0.0
                    elif feat in ["std_x", "std_z"]:
                        pitch_data[feat] = 0.5
                    elif feat == "pct_in_zone":
                        pitch_data[feat] = 0.5
                    elif feat == "pct_vs_RHB":
                        pitch_data[feat] = 0.5
                    else:
                        pitch_data[feat] = None
            arsenal_rows.append(pitch_data)

    if not arsenal_rows:
        return pd.DataFrame(columns=["pitch_type"] + FEATURE_SUFFIXES)

    arsenal_df = pd.DataFrame(arsenal_rows)
    arsenal_df = arsenal_df.sort_values("usage", ascending=False).reset_index(drop=True)

    return arsenal_df, p_throws


def normalize_usage(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0 or "usage" not in df.columns:
        return df
    s = df["usage"].sum()
    if pd.notna(s) and s > 0:
        df = df.copy()
        df["usage"] = df["usage"] / s
    return df


def default_empty_arsenal() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "pitch_type", "usage", "velo", "HB", "IVB", "VAA", "HAA", "spin", "ext",
        "rel_x", "rel_z", "mean_x", "mean_z", "std_x", "std_z", "pct_in_zone", "pct_vs_RHB"
    ])


def ensure_plot_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Plotly won't render points with NaN coords
    df["HB"] = pd.to_numeric(df["HB"], errors="coerce").fillna(0.0)
    df["IVB"] = pd.to_numeric(df["IVB"], errors="coerce").fillna(0.0)
    df["usage"] = pd.to_numeric(df["usage"], errors="coerce").fillna(0.05)

    # Location columns
    df["mean_x"] = pd.to_numeric(df.get("mean_x", 0.0), errors="coerce").fillna(0.0)
    df["mean_z"] = pd.to_numeric(df.get("mean_z", 2.5), errors="coerce").fillna(2.5)
    df["std_x"] = pd.to_numeric(df.get("std_x", 0.5), errors="coerce").fillna(0.5)
    df["std_z"] = pd.to_numeric(df.get("std_z", 0.5), errors="coerce").fillna(0.5)

    # Ensure visible marker sizes (usage can be tiny)
    df["plot_size"] = np.clip(df["usage"] * 80, 12, 45)
    # Location plot size based on average of std_x and std_z
    df["location_size"] = np.clip((df["std_x"] + df["std_z"]) * 30, 10, 50)
    df["idx"] = np.arange(len(df))
    return df


def create_strike_zone_plot(df_plot: pd.DataFrame) -> go.Figure:
    """Create a location plot with strike zone and home plate."""
    fig = go.Figure()

    # Draw strike zone (rectangle)
    fig.add_shape(
        type="rect",
        x0=ZONE_LEFT, y0=ZONE_BOTTOM,
        x1=ZONE_RIGHT, y1=ZONE_TOP,
        line=dict(color="black", width=2),
        fillcolor="rgba(200, 200, 200, 0.2)",
    )

    # Draw home plate (pentagon)
    plate_half = PLATE_WIDTH / 2
    plate_points = [
        (-plate_half, 0),
        (plate_half, 0),
        (plate_half, 0.25),
        (0, 0.5),
        (-plate_half, 0.25),
        (-plate_half, 0),  # close the shape
    ]
    fig.add_trace(go.Scatter(
        x=[p[0] for p in plate_points],
        y=[p[1] for p in plate_points],
        mode="lines",
        fill="toself",
        fillcolor="white",
        line=dict(color="black", width=2),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Add pitch location points
    colors = px.colors.qualitative.Plotly
    pitch_types = df_plot["pitch_type"].unique()
    color_map = {pt: colors[i % len(colors)] for i, pt in enumerate(pitch_types)}

    for pitch_type in pitch_types:
        mask = df_plot["pitch_type"] == pitch_type
        sub = df_plot[mask]
        if len(sub) == 0:
            continue

        fig.add_trace(go.Scatter(
            x=sub["mean_x"],
            y=sub["mean_z"],
            mode="markers+text",
            name=pitch_type,
            text=sub["pitch_type"],
            textposition="top center",
            marker=dict(
                size=sub["location_size"],
                color=color_map[pitch_type],
                opacity=0.7,
                line=dict(width=1, color="white"),
            ),
            customdata=np.stack([sub["std_x"], sub["std_z"], sub["pct_in_zone"]], axis=-1) if "pct_in_zone" in sub.columns else np.stack([sub["std_x"], sub["std_z"]], axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Mean X: %{x:.2f} ft<br>"
                "Mean Z: %{y:.2f} ft<br>"
                "Std X: %{customdata[0]:.2f}<br>"
                "Std Z: %{customdata[1]:.2f}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Pitch Location (Catcher's View)",
        xaxis_title="Horizontal (ft, + = catcher's right)",
        yaxis_title="Height (ft)",
        height=450,
        showlegend=True,
        xaxis=dict(
            range=[-2.5, 2.5],
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="gray",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[-0.2, 5],
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="gray",
        ),
    )

    return fig


@st.cache_data(show_spinner=False)
def build_pitcher_map_from_ids(pitcher_ids: list[int]) -> pd.DataFrame:
    """
    Build mapping pitcher (MLBAM id) -> player_name using pybaseball reverse lookup.
    Cached by Streamlit so it won't re-run unless inputs change.
    """
    if len(pitcher_ids) == 0:
        return pd.DataFrame(columns=["pitcher", "player_name"])

    lu = playerid_reverse_lookup(pitcher_ids, key_type="mlbam")

    # Construct a display name
    lu["player_name"] = (
        lu["name_first"].fillna("").astype(str).str.strip()
        + " "
        + lu["name_last"].fillna("").astype(str).str.strip()
    ).str.strip()

    out = lu[["key_mlbam", "player_name"]].rename(columns={"key_mlbam": "pitcher"}).copy()
    out["pitcher"] = pd.to_numeric(out["pitcher"], errors="coerce").astype("Int64")
    out["player_name"] = out["player_name"].astype("string")

    # Fill unknowns defensively
    out["player_name"] = out["player_name"].fillna("Unknown")

    return out.drop_duplicates(subset=["pitcher"])


def get_or_create_pitcher_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads pitcher_id_to_name.parquet if present.
    Otherwise builds it using playerid_reverse_lookup and writes it to disk.
    """
    if PITCHER_MAP_PATH.exists():
        m = pd.read_parquet(PITCHER_MAP_PATH)
        m["pitcher"] = pd.to_numeric(m["pitcher"], errors="coerce").astype("Int64")
        m["player_name"] = m["player_name"].astype("string")
        return m.drop_duplicates(subset=["pitcher"])

    ids = sorted(df["pitcher"].dropna().astype(int).unique().tolist())
    m = build_pitcher_map_from_ids(ids)

    # Save for future fast startup
    PITCHER_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(PITCHER_MAP_PATH, index=False)

    return m


# ---------- Load data ----------
if not DATA_PATH.exists():
    st.error(f"Could not find parquet at: {DATA_PATH}")
    st.stop()

data = load_processed(str(DATA_PATH))

# Build/load pitcher name map
with st.sidebar:
    st.header("Data")

    # Optional: refresh mapping button (useful if your data grows)
    refresh_map = st.button("Refresh pitcher names (rebuild map)")

if refresh_map and PITCHER_MAP_PATH.exists():
    try:
        PITCHER_MAP_PATH.unlink()
        st.sidebar.success("Deleted pitcher_id_to_name.parquet. Reloading will rebuild it.")
    except Exception as e:
        st.sidebar.error(f"Could not delete mapping file: {e}")

pitcher_map = get_or_create_pitcher_map(data)

# Attach names to data for dropdown building (doesn't change modeling)
data_named = data.merge(pitcher_map, on="pitcher", how="left")
data_named["player_name"] = data_named["player_name"].fillna("Unknown")

# Dropdown index
pitcher_index = (
    data_named[["pitcher", "player_name"]]
    .drop_duplicates()
    .sort_values(["player_name", "pitcher"])
)
pitcher_index["label"] = pitcher_index["player_name"].astype(str) + " (" + pitcher_index["pitcher"].astype(str) + ")"
labels = pitcher_index["label"].tolist()
label_to_id = dict(zip(pitcher_index["label"], pitcher_index["pitcher"].astype(int)))

seasons = sorted([int(x) for x in data_named["season"].dropna().unique().tolist()])


# ---------- Session state ----------
if "arsenal" not in st.session_state:
    st.session_state.arsenal = default_empty_arsenal()
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0
if "editor_version" not in st.session_state:
    st.session_state.editor_version = 0
if "p_throws" not in st.session_state:
    st.session_state.p_throws = "R"


# ---------- Sidebar: mode + loader ----------
st.sidebar.header("Load Arsenal")
mode = st.sidebar.radio("Mode", ["Load MLB pitcher-season", "New arsenal"], index=0)

# Initialize variables that will be used later
pitcher_id = None
chosen_season = None

if mode == "New arsenal":
    st.sidebar.divider()
    st.session_state.p_throws = st.sidebar.radio(
        "Pitcher Handedness",
        ["R", "L"],
        index=0 if st.session_state.p_throws == "R" else 1,
        horizontal=True,
    )
    st.sidebar.divider()

    if st.sidebar.button("Start empty arsenal"):
        st.session_state.arsenal = default_empty_arsenal()
        st.session_state.selected_idx = 0
        st.rerun()

    if st.sidebar.button("Load sample defaults"):
        sample = pd.DataFrame([
            {"pitch_type": "FF", "usage": 0.5, "velo": 94, "HB": -5, "IVB": 15, "VAA": -4.5, "HAA": 0, "spin": 2300, "ext": 6.5, "rel_x": -1.5, "rel_z": 5.8, "mean_x": 0.0, "mean_z": 2.8, "std_x": 0.6, "std_z": 0.5, "pct_in_zone": 0.55, "pct_vs_RHB": 0.5},
            {"pitch_type": "SL", "usage": 0.3, "velo": 85, "HB": 6, "IVB": 2, "VAA": -6.0, "HAA": 1, "spin": 2500, "ext": 6.3, "rel_x": -1.5, "rel_z": 5.8, "mean_x": 0.3, "mean_z": 2.2, "std_x": 0.7, "std_z": 0.6, "pct_in_zone": 0.40, "pct_vs_RHB": 0.5},
            {"pitch_type": "CH", "usage": 0.2, "velo": 86, "HB": -10, "IVB": 9, "VAA": -5.0, "HAA": -0.5, "spin": 1800, "ext": 6.4, "rel_x": -1.5, "rel_z": 5.8, "mean_x": -0.2, "mean_z": 2.0, "std_x": 0.5, "std_z": 0.5, "pct_in_zone": 0.35, "pct_vs_RHB": 0.5},
        ])
        st.session_state.arsenal = normalize_usage(sample)
        st.session_state.selected_idx = 0
        st.rerun()

else:  # Load MLB pitcher-season mode
    chosen_label = st.sidebar.selectbox("Pitcher", labels)
    chosen_season = st.sidebar.selectbox("Season", seasons, index=len(seasons) - 1)
    pitcher_id = label_to_id[chosen_label]

    # Load button is now inside the correct scope
    if st.sidebar.button("Load this arsenal"):
        result = extract_arsenal_from_wide(data, pitcher_id, chosen_season)
        if isinstance(result, tuple):
            arsenal_df, p_throws_val = result
            if p_throws_val in ["R", "L"]:
                st.session_state.p_throws = p_throws_val
        else:
            arsenal_df = result

        arsenal_df = normalize_usage(arsenal_df)
        st.session_state.arsenal = arsenal_df
        st.session_state.selected_idx = 0

        # Force data_editor to reinitialize
        st.session_state.editor_version += 1

        st.rerun()


# ---------- Main layout ----------
left, right = st.columns([1.25, 1])

with left:
    st.subheader("Arsenal Table")

    arsenal = st.session_state.arsenal.copy()

    editor_key = f"arsenal_editor_{st.session_state.editor_version}"

    edited = st.data_editor(
        st.session_state.arsenal,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "pitch_type": st.column_config.SelectboxColumn("Pitch", options=PITCH_TYPES, width="small"),
            "usage": st.column_config.NumberColumn("Usage", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
            "velo": st.column_config.NumberColumn("Velo", min_value=60.0, max_value=110.0, step=0.1, format="%.1f"),
            "HB": st.column_config.NumberColumn("HB (in)", min_value=-25.0, max_value=25.0, step=0.5, format="%.1f"),
            "IVB": st.column_config.NumberColumn("IVB (in)", min_value=-25.0, max_value=25.0, step=0.5, format="%.1f"),
            "VAA": st.column_config.NumberColumn("VAA", min_value=-10.0, max_value=0.0, step=0.1, format="%.1f"),
            "HAA": st.column_config.NumberColumn("HAA", min_value=-5.0, max_value=5.0, step=0.1, format="%.1f"),
            "spin": st.column_config.NumberColumn("Spin", min_value=1000, max_value=3500, step=50, format="%d"),
            "ext": st.column_config.NumberColumn("Ext (ft)", min_value=4.0, max_value=8.0, step=0.1, format="%.1f"),
            "rel_x": st.column_config.NumberColumn("Rel X", min_value=-4.0, max_value=4.0, step=0.1, format="%.1f"),
            "rel_z": st.column_config.NumberColumn("Rel Z", min_value=4.0, max_value=7.0, step=0.1, format="%.1f"),
            "mean_x": st.column_config.NumberColumn("Loc X", min_value=-2.0, max_value=2.0, step=0.05, format="%.2f", help="Mean horizontal location (ft)"),
            "mean_z": st.column_config.NumberColumn("Loc Z", min_value=0.5, max_value=5.0, step=0.05, format="%.2f", help="Mean vertical location (ft)"),
            "std_x": st.column_config.NumberColumn("SD X", min_value=0.1, max_value=2.0, step=0.05, format="%.2f", help="Horizontal command (std dev)"),
            "std_z": st.column_config.NumberColumn("SD Z", min_value=0.1, max_value=2.0, step=0.05, format="%.2f", help="Vertical command (std dev)"),
            "pct_in_zone": st.column_config.NumberColumn("Zone%", min_value=0.0, max_value=1.0, step=0.01, format="%.0%%", help="Percentage in strike zone"),
            "pct_vs_RHB": st.column_config.NumberColumn("vs RHB%", min_value=0.0, max_value=1.0, step=0.01, format="%.0%%", help="Pct thrown vs right-handed batters"),
        },
        key=editor_key,
    )

    if len(edited) > 0:
        if st.checkbox("Auto-normalize usage to sum to 1", value=True):
            edited = normalize_usage(edited)

    st.session_state.arsenal = edited


with right:
    st.subheader("Movement Plot (HB vs IVB)")

    df_plot = ensure_plot_cols(st.session_state.arsenal)

    if len(df_plot) > 0:
        # highlight selected point
        sel = int(np.clip(st.session_state.selected_idx, 0, max(len(df_plot) - 1, 0)))
        df_plot["selected"] = (df_plot["idx"] == sel)

        fig = px.scatter(
            df_plot,
            x="HB",
            y="IVB",
            text="pitch_type",
            color="pitch_type",
            size="plot_size",
            hover_data=["usage", "velo", "VAA", "spin", "ext"],
            title="Pitch Movement"
        )
        fig.update_traces(textposition="top center", marker=dict(sizemode="diameter"))
        fig.update_layout(height=400, showlegend=True)
        fig.update_xaxes(zeroline=True, title="Horizontal Break (in)", range=[-25, 25])
        fig.update_yaxes(zeroline=True, title="Induced Vertical Break (in)", range=[-25, 25])

        st.plotly_chart(fig, use_container_width=True)

        # Location plot with strike zone
        st.subheader("Location Plot (Strike Zone)")
        location_fig = create_strike_zone_plot(df_plot)
        st.plotly_chart(location_fig, use_container_width=True)

        # Quick edit for selected pitch
        st.divider()
        sel = int(np.clip(st.session_state.selected_idx, 0, len(df_plot) - 1))
        pitch_options = [f"{i}: {df_plot.loc[i, 'pitch_type']}" for i in range(len(df_plot))]
        selected_option = st.selectbox(
            "Select pitch to edit",
            pitch_options,
            index=sel,
        )
        sel = int(selected_option.split(":")[0])
        st.session_state.selected_idx = sel

        st.markdown(f"**Editing:** `{df_plot.loc[sel, 'pitch_type']}`")

        col1, col2 = st.columns(2)
        with col1:
            hb = st.slider("HB (inches)", -25.0, 25.0, float(df_plot.loc[sel, "HB"]), 0.5)
            ivb = st.slider("IVB (inches)", -25.0, 25.0, float(df_plot.loc[sel, "IVB"]), 0.5)
        with col2:
            mean_x = st.slider("Location X (ft)", -2.0, 2.0, float(df_plot.loc[sel, "mean_x"]), 0.05)
            mean_z = st.slider("Location Z (ft)", 0.5, 5.0, float(df_plot.loc[sel, "mean_z"]), 0.05)

        if st.button("Apply changes to selected pitch"):
            updated = st.session_state.arsenal.copy()
            if len(updated) > sel:
                updated.loc[sel, "HB"] = hb
                updated.loc[sel, "IVB"] = ivb
                updated.loc[sel, "mean_x"] = mean_x
                updated.loc[sel, "mean_z"] = mean_z
            st.session_state.arsenal = updated
            st.rerun()
    else:
        st.info("No pitches in arsenal. Load a pitcher or add pitches manually to see the plot.")


st.divider()
st.subheader("Model Features Summary")
st.caption("These are all the features that go into the pitch models for predicting whiff rate.")

# Show pitcher handedness
col_info1, col_info2 = st.columns([1, 4])
with col_info1:
    st.metric("Pitcher Throws", st.session_state.p_throws)

# Display a cleaner view of the arsenal with all model features
if len(st.session_state.arsenal) > 0:
    display_cols = [
        "pitch_type", "usage", "velo", "HB", "IVB", "VAA", "HAA", "spin", "ext",
        "rel_x", "rel_z", "mean_x", "mean_z", "std_x", "std_z", "pct_in_zone", "pct_vs_RHB"
    ]
    display_df = st.session_state.arsenal[[c for c in display_cols if c in st.session_state.arsenal.columns]].copy()
    st.dataframe(display_df, use_container_width=True)
else:
    st.info("Load or create an arsenal to see the model features.")


