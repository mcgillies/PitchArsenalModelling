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

# Your parquet doesn't have player_name/spin_axis; we won't require them
REQUIRED_COLS = [
    "pitcher", "season", "pitch_type",
    "usage", "velo", "HB", "IVB", "VAA", "HAA", "spin", "ext", "rel_x", "rel_z",
]

PITCH_TYPES = ["FF", "SI", "FC", "FT", "SL", "CU", "CH", "FS", "KC"]


# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_processed(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}")

    # Ensure numeric columns are numeric (prevents invisible points)
    num_cols = ["usage", "velo", "HB", "IVB", "VAA", "HAA", "spin", "ext", "rel_x", "rel_z"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce").astype("Int64")
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["pitch_type"] = df["pitch_type"].astype("string")

    return df


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
        "pitch_type", "usage", "velo", "HB", "IVB", "VAA", "HAA", "spin", "ext", "rel_x", "rel_z"
    ])


def ensure_plot_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Plotly won't render points with NaN coords
    df["HB"] = pd.to_numeric(df["HB"], errors="coerce").fillna(0.0)
    df["IVB"] = pd.to_numeric(df["IVB"], errors="coerce").fillna(0.0)
    df["usage"] = pd.to_numeric(df["usage"], errors="coerce").fillna(0.05)

    # Ensure visible marker sizes (usage can be tiny)
    df["plot_size"] = np.clip(df["usage"] * 80, 12, 45)
    df["idx"] = np.arange(len(df))
    return df


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


# ---------- Sidebar: mode + loader ----------
st.sidebar.header("Load Arsenal")
mode = st.sidebar.radio("Mode", ["Load MLB pitcher-season", "New arsenal"], index=0)

# Initialize variables that will be used later
pitcher_id = None
chosen_season = None

if mode == "New arsenal":
    if st.sidebar.button("Start empty arsenal"):
        st.session_state.arsenal = default_empty_arsenal()
        st.session_state.selected_idx = 0
        st.rerun()

    if st.sidebar.button("Load sample defaults"):
        sample = pd.DataFrame([
            {"pitch_type": "FF", "usage": 0.5, "velo": 94, "HB": -5, "IVB": 15, "VAA": -4.5, "HAA": 0, "spin": 2300, "ext": 6.5, "rel_x": -1.5, "rel_z": 5.8},
            {"pitch_type": "SL", "usage": 0.3, "velo": 85, "HB": 6, "IVB": 2, "VAA": -6.0, "HAA": 1, "spin": 2500, "ext": 6.3, "rel_x": -1.5, "rel_z": 5.8},
            {"pitch_type": "CH", "usage": 0.2, "velo": 86, "HB": -10, "IVB": 9, "VAA": -5.0, "HAA": -0.5, "spin": 1800, "ext": 6.4, "rel_x": -1.5, "rel_z": 5.8},
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
        sub = data_named[(data_named["pitcher"] == pitcher_id) & (data_named["season"] == chosen_season)].copy()

        arsenal_cols = ["pitch_type","usage","velo","HB","IVB","VAA","HAA","spin","ext","rel_x","rel_z"]
        sub = sub[arsenal_cols].dropna(subset=["pitch_type"]).sort_values("usage", ascending=False)
        sub = normalize_usage(sub.reset_index(drop=True))

        st.session_state.arsenal = sub
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
        use_container_width=True,  # Changed from width="stretch"
        column_config={
            "pitch_type": st.column_config.SelectboxColumn("pitch_type", options=PITCH_TYPES),
            "usage": st.column_config.NumberColumn("usage", min_value=0.0, max_value=1.0, step=0.01),
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
            size="plot_size",
            hover_data=["usage", "velo", "VAA", "spin", "ext"],
            symbol="selected",
            title="Click a point to select it (then adjust HB/IVB below)"
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=520)
        fig.update_xaxes(zeroline=True)
        fig.update_yaxes(zeroline=True)

        if HAS_PLOTLY_EVENTS:
            selected = plotly_events(fig, click_event=True, hover_event=False, select_event=False)
            if selected:
                st.session_state.selected_idx = int(selected[0]["pointIndex"])
                sel = st.session_state.selected_idx
        else:
            st.plotly_chart(fig, use_container_width=True)
            st.info("Install click-to-select support: pip install streamlit-plotly-events")

        sel = int(np.clip(st.session_state.selected_idx, 0, len(df_plot) - 1))
        st.markdown(f"**Selected pitch:** `{df_plot.loc[sel, 'pitch_type']}` (row {sel})")

        hb = st.slider("HB (inches)", -25.0, 25.0, float(df_plot.loc[sel, "HB"]), 0.5)
        ivb = st.slider("IVB (inches)", -25.0, 25.0, float(df_plot.loc[sel, "IVB"]), 0.5)

        if st.button("Apply HB/IVB to selected pitch"):
            updated = st.session_state.arsenal.copy()
            if len(updated) > sel:
                updated.loc[sel, "HB"] = hb
                updated.loc[sel, "IVB"] = ivb
            st.session_state.arsenal = updated
            st.rerun()
    else:
        st.info("No pitches in arsenal. Load a pitcher or add pitches manually to see the plot.")


st.divider()
st.subheader("Predictions (next step)")
st.write("Wire your model inference here: per-pitch whiff/CSW + arsenal aggregate.")
st.dataframe(st.session_state.arsenal, use_container_width=True)


