"""Stage 1B predictor inspector: browse Stage 1B predictions alongside candlestick charts.

Usage:
    streamlit run scripts/inspect_stage1b.py
    streamlit run scripts/inspect_stage1b.py -- --predictions path/to/predictions_with_proximity.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_PAST_BARS = 48
DEFAULT_FUTURE_BARS = 8
HORIZON_BARS = {"h4": 4, "h8": 8, "h16": 16, "h32": 32, "h48": 48}
DISPLAY_PAST_OPTIONS = {
    "Short window · 12h / 48 bars": 48,
    "Long window · 72h / 288 bars": 288,
}
DISPLAY_FUTURE_OPTIONS = {
    "Selected label horizon": None,
    "H8 · 2h / 8 bars": 8,
    "H16 · 4h / 16 bars": 16,
    "H32 · 8h / 32 bars": 32,
    "H48 · 12h / 48 bars": 48,
    "Forward 72h / 288 bars": 288,
}

LABEL_NAMES = {0: "intact", 1: "bullish", 2: "bearish"}
LABEL_COLORS = {"intact": "#9e9e9e", "bullish": "#26a69a", "bearish": "#ef5350", "none": "#9e9e9e"}
LABEL_EMOJI = {"intact": "⬜ intact", "bullish": "🟢 bullish", "bearish": "🔴 bearish", "none": "⬜ none"}

SEGMENT_OPTIONS = {
    "All": None,
    "Correct": "correct",
    "False positive break": "false_positive_break",
    "False negative break": "false_negative_break",
    "Wrong direction": "wrong_direction",
    "High confidence wrong (≥0.90)": "high_confidence_wrong",
}

CONF_BUCKETS = {
    "All confidence": (0.0, 1.0),
    "Low  (0.50–0.70)": (0.50, 0.70),
    "Medium (0.70–0.90)": (0.70, 0.90),
    "High (0.90–1.00)": (0.90, 1.01),
}

DEFAULT_PREDICTIONS = (
    "data/stage1b/binance/15m/checkpoints/"
    "stage1b_h16_joint_event_sequence_v1_z_fused_proximity/test_joint_event_predictions.csv"
)
DEFAULT_LABELS = (
    "data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/"
    "analysis/stage1b_forward_labels_test_h4_h8_h16_v2.parquet"
)
RAW_DATA_ROOT = Path("data/raw/binance")
HORIZON_OPTIONS = ["h4", "h8", "h16", "h32", "h48"]


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_predictions(csv_path: str, label_path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "true_joint_event" in df.columns:
        df = normalize_joint_predictions(df)
    if label_path and Path(label_path).exists():
        labels = pd.read_parquet(label_path)
        merge_cols = ["index", "symbol", "timestamp"]
        if all(col in df.columns for col in merge_cols) and all(col in labels.columns for col in merge_cols):
            label_cols = [
                col
                for col in labels.columns
                if col not in df.columns or col in merge_cols
            ]
            df = df.merge(labels[label_cols], on=merge_cols, how="left")
    df["anchor_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_localize(None)
    if "is_correct" not in df.columns:
        df["is_correct"] = df.get("is_joint_correct", False)
    if "break_confidence" not in df.columns:
        df["break_confidence"] = df.get("joint_confidence", 0.0)
    if "p_break" not in df.columns:
        df["p_break"] = 1.0 - df["p_no_event"] if "p_no_event" in df.columns else df["break_confidence"]
    if "pred_direction" not in df.columns:
        df["pred_direction"] = df.get("pred_event_direction", df.get("pred_label", "none"))
    if "direction_confidence" not in df.columns:
        df["direction_confidence"] = df.get("joint_confidence", 0.0)
    df["high_confidence_wrong"] = (~df["is_correct"].astype(bool)) & (df["break_confidence"] >= 0.90)
    return df


def normalize_joint_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["true_label"] = df["true_event_direction"].fillna("none").astype(str)
    df["pred_label"] = df["pred_event_direction"].fillna("none").astype(str)
    df["is_correct"] = df["is_joint_correct"].astype(bool)
    df["break_confidence"] = df["joint_confidence"].astype(float)
    df["p_break"] = 1.0 - df["p_no_event"].astype(float)
    df["pred_direction"] = df["pred_event_direction"].astype(str)
    df["direction_confidence"] = df["joint_confidence"].astype(float)
    df["candidate_strength"] = df["joint_confidence"].astype(float)
    df["candidate_gate"] = (
        (df["pred_joint_event"] != "no_event")
        & (
            (df["joint_confidence"] >= 0.50)
            | ((df["pred_event_type"] == "reversal") & (df["joint_confidence"] >= 0.45))
        )
    )

    def error_type(row: pd.Series) -> str:
        true_joint = str(row["true_joint_event"])
        pred_joint = str(row["pred_joint_event"])
        if true_joint == pred_joint:
            return "correct"
        if true_joint == "no_event" and pred_joint != "no_event":
            return "false_positive_break"
        if true_joint != "no_event" and pred_joint == "no_event":
            return "false_negative_break"
        if str(row["true_event_type"]) == str(row["pred_event_type"]):
            return "wrong_direction"
        return "wrong_event"

    df["error_type"] = df.apply(error_type, axis=1)
    return df


@st.cache_data
def load_ohlcv(symbol: str) -> pd.DataFrame:
    path = RAW_DATA_ROOT / f"{symbol}.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def get_window(
    ohlcv: pd.DataFrame,
    anchor_dt: pd.Timestamp,
    past_bars: int,
    future_bars: int,
) -> pd.DataFrame | None:
    matches = ohlcv.index[ohlcv["timestamp"] == anchor_dt].tolist()
    if not matches:
        return None
    anchor_idx = matches[0]
    start = anchor_idx - past_bars + 1
    end = anchor_idx + future_bars + 1
    if start < 0 or end > len(ohlcv):
        return None
    return ohlcv.iloc[start:end].reset_index(drop=True)


# ── Chart ─────────────────────────────────────────────────────────────────────

def build_chart(
    window_df: pd.DataFrame,
    row: pd.Series,
    selected_horizon: str = "h8",
    past_bars: int = DEFAULT_PAST_BARS,
    future_bars: int = DEFAULT_FUTURE_BARS,
) -> go.Figure:
    opens  = window_df["open"].tolist()
    highs  = window_df["high"].tolist()
    lows   = window_df["low"].tolist()
    closes = window_df["close"].tolist()
    n = len(closes)
    xs = list(range(n))

    anchor_x = past_bars - 1
    future_start = past_bars
    horizon_bars = HORIZON_BARS.get(selected_horizon, DEFAULT_FUTURE_BARS)

    true_label = str(row["true_label"])
    true_event_type = str(row.get("true_event_type", ""))
    future_color_hex = LABEL_COLORS.get(true_label, "#9e9e9e")
    if true_event_type == "reversal":
        future_color_hex = "#ab47bc"
    elif true_event_type == "continuation":
        future_color_hex = "#26a69a" if true_label == "bullish" else "#ef5350"

    def hex_rgb(h: str) -> str:
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"

    fig = go.Figure()

    # Background zones
    fig.add_vrect(
        x0=-0.5, x1=anchor_x + 0.5,
        fillcolor="rgba(100,100,100,0.07)", line_width=0,
        annotation_text=f"past ({past_bars} bars)", annotation_position="top left",
        annotation_font_size=10, annotation_font_color="#666",
    )
    fig.add_vrect(
        x0=future_start - 0.5, x1=n - 0.5,
        fillcolor=f"rgba({hex_rgb(future_color_hex)},0.12)", line_width=0,
        annotation_text=f"future ({future_bars} bars)", annotation_position="top right",
        annotation_font_size=10, annotation_font_color=future_color_hex,
    )
    if horizon_bars < future_bars:
        horizon_end = past_bars + horizon_bars - 0.5
        fig.add_vline(
            x=horizon_end,
            line_dash="dot",
            line_color="#777",
            line_width=1.0,
            annotation_text=selected_horizon.upper(),
            annotation_position="top",
            annotation_font_size=10,
            annotation_font_color="#999",
        )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=xs, open=opens, high=highs, low=lows, close=closes,
        name="price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
    ))

    # Anchor divider
    fig.add_vline(
        x=anchor_x + 0.5,
        line_dash="dash", line_color="#aaaaaa", line_width=1.5,
    )

    time_to_break = row.get(f"{selected_horizon}_time_to_break")
    if pd.notna(time_to_break) and int(time_to_break) > 0:
        break_x = anchor_x + int(time_to_break)
        if break_x <= n - 1:
            fig.add_vline(
                x=break_x,
                line_dash="dot",
                line_color="#42a5f5",
                line_width=1.2,
                annotation_text="confirm",
                annotation_position="top",
                annotation_font_size=10,
                annotation_font_color="#42a5f5",
            )

    # Break levels
    bull_level = row.get("bull_level")
    bear_level = row.get("bear_level")

    if pd.notna(bull_level):
        fig.add_hline(
            y=bull_level, line_dash="dash", line_color="#26a69a", line_width=1.5,
            annotation_text=f"bull level {bull_level:.4f}",
            annotation_position="right",
            annotation_font_size=10, annotation_font_color="#26a69a",
        )
    if pd.notna(bear_level):
        fig.add_hline(
            y=bear_level, line_dash="dash", line_color="#ef5350", line_width=1.5,
            annotation_text=f"bear level {bear_level:.4f}",
            annotation_position="right",
            annotation_font_size=10, annotation_font_color="#ef5350",
        )

    # Title: error type
    error_type = str(row.get("error_type", ""))
    error_colors = {
        "correct": "#26a69a",
        "false_positive_break": "#ff9800",
        "false_negative_break": "#2196f3",
        "wrong_direction": "#ef5350",
        "wrong_event": "#ab47bc",
    }
    title_color = error_colors.get(error_type, "#cccccc")
    semantic = str(row.get(f"{selected_horizon}_break_semantic_label", ""))
    outcome = str(row.get(f"{selected_horizon}_post_break_outcome", ""))
    title_text = error_type.replace("_", " ").upper()
    if "pred_joint_event" in row.index:
        title_text = f"{title_text} · pred {str(row['pred_joint_event']).replace('_', ' ')}"
    if semantic and semantic != "nan":
        title_text = f"{title_text} · {semantic.replace('_', ' ')}"
    if outcome and outcome not in {"nan", semantic}:
        title_text = f"{title_text} · {outcome.replace('_', ' ')}"

    fig.update_layout(
        title=dict(
            text=f"<b style='color:{title_color}'>{title_text}</b>",
            font=dict(size=18), x=0.5,
        ),
        xaxis_title="bar index",
        yaxis_title="price",
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=40, r=140, t=60, b=40),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f23",
        font=dict(color="#cccccc"),
        xaxis=dict(gridcolor="#2a2a3e", tickfont=dict(color="#888")),
        yaxis=dict(gridcolor="#2a2a3e", tickfont=dict(color="#888")),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
    )
    return fig


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Stage 1B Inspector",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        [data-testid="stToolbar"] {display: none;}
        [data-testid="stDecoration"] {display: none;}
        [data-testid="stDeployButton"] {display: none;}
        header {display: none !important; height: 0 !important;}
        [data-testid="stHeader"] {
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
        }
        [data-testid="stSidebarCollapsedControl"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
        [data-testid="stSidebar"] {
            min-width: 19rem;
            max-width: 19rem;
            width: 19rem;
            transform: translateX(0) !important;
            visibility: visible !important;
        }
        .block-container {
            padding-top: 0.65rem;
            padding-bottom: 1.2rem;
            padding-left: 1.0rem;
            padding-right: 1.0rem;
            max-width: 100%;
        }
        [data-testid="stSidebar"] h1 {
            font-size: 1.18rem;
            line-height: 1.25;
            margin-bottom: 0.2rem;
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-size: 0.95rem;
            margin-top: 0.65rem;
        }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.025);
            border: 1px solid rgba(255, 255, 255, 0.055);
            border-radius: 6px;
            padding: 0.35rem 0.45rem;
        }
        [data-testid="stMetricLabel"] {font-size: 0.68rem;}
        [data-testid="stMetricValue"] {font-size: 0.92rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Stage 1B\nPredictor")
        st.caption("Fresh break + continuation label audit")
        st.divider()
        st.header("Navigation")
        nav_slot = st.container()
        st.header("Data")

        cli_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
        default_csv = DEFAULT_PREDICTIONS
        if "--predictions" in cli_args:
            default_csv = cli_args[cli_args.index("--predictions") + 1]
        default_label_path = DEFAULT_LABELS
        if "--labels" in cli_args:
            default_label_path = cli_args[cli_args.index("--labels") + 1]

        csv_path = st.text_input("Predictions CSV", value=default_csv)
        if not Path(csv_path).exists():
            st.error(f"File not found: {csv_path}")
            st.stop()
        label_path = st.text_input("Label context parquet", value=default_label_path)
        if label_path and not Path(label_path).exists():
            st.warning(f"Label context not found, continuing without merge: {label_path}")
            label_path = ""

        df = load_predictions(csv_path, label_path or None)

        st.divider()
        st.header("Filters")

        symbols_available = sorted(df["symbol"].unique())
        symbol_filter = st.multiselect("Symbol", options=symbols_available, default=symbols_available)

        segment_label = st.selectbox("Segment", options=list(SEGMENT_OPTIONS.keys()))
        conf_label = st.selectbox("Confidence", options=list(CONF_BUCKETS.keys()))
        available_horizons = [
            horizon
            for horizon in HORIZON_OPTIONS
            if f"{horizon}_break_anchor_status" in df.columns or f"{horizon}_break_semantic_label" in df.columns
        ]
        selected_horizon = st.selectbox(
            "Label horizon",
            options=available_horizons or ["h8"],
            index=(available_horizons or ["h8"]).index("h8") if "h8" in (available_horizons or ["h8"]) else 0,
        )

        st.header("Plot range")
        past_label = st.selectbox("Past bars", options=list(DISPLAY_PAST_OPTIONS.keys()))
        future_label = st.selectbox("Future bars", options=list(DISPLAY_FUTURE_OPTIONS.keys()))
        display_past_bars = DISPLAY_PAST_OPTIONS[past_label]
        display_future_bars = DISPLAY_FUTURE_OPTIONS[future_label]
        if display_future_bars is None:
            display_future_bars = HORIZON_BARS.get(selected_horizon, DEFAULT_FUTURE_BARS)

        true_labels = sorted(df["true_label"].dropna().astype(str).unique())
        pred_labels = sorted(df["pred_label"].dropna().astype(str).unique())
        true_filter = st.multiselect("True label", options=true_labels, default=true_labels)
        pred_filter = st.multiselect("Pred label", options=pred_labels, default=pred_labels)
        if "candidate_gate" in df.columns:
            candidate_filter = st.multiselect(
                "Candidate gate",
                options=["candidate", "not_candidate"],
                default=["candidate", "not_candidate"],
            )
        else:
            candidate_filter = None

        status_col = f"{selected_horizon}_break_anchor_status"
        semantic_col = f"{selected_horizon}_break_semantic_label"
        fresh_col = f"{selected_horizon}_fresh_break_direction"
        outcome_col = f"{selected_horizon}_post_break_outcome"
        dominant_col = f"{selected_horizon}_dominant_forward_direction"
        status_filter = None
        semantic_filter = None
        fresh_filter = None
        outcome_filter = None
        dominant_filter = None
        if status_col in df.columns:
            status_values = sorted(df[status_col].dropna().astype(str).unique())
            status_filter = st.multiselect("Anchor status", options=status_values, default=status_values)
        if semantic_col in df.columns:
            semantic_values = sorted(df[semantic_col].dropna().astype(str).unique())
            semantic_filter = st.multiselect("Semantic label", options=semantic_values, default=semantic_values)
        if fresh_col in df.columns:
            fresh_values = sorted(df[fresh_col].dropna().astype(str).unique())
            fresh_filter = st.multiselect("Fresh direction", options=fresh_values, default=fresh_values)
        if outcome_col in df.columns:
            outcome_values = sorted(df[outcome_col].dropna().astype(str).unique())
            outcome_filter = st.multiselect("Post-break outcome", options=outcome_values, default=outcome_values)
        if dominant_col in df.columns:
            dominant_values = sorted(df[dominant_col].dropna().astype(str).unique())
            dominant_filter = st.multiselect("Dominant direction", options=dominant_values, default=dominant_values)

        random_mode = st.checkbox("Random jump", value=False)

    # ── Filter ────────────────────────────────────────────────────────────────
    mask = df["symbol"].isin(symbol_filter)
    mask &= df["true_label"].astype(str).isin(true_filter)
    mask &= df["pred_label"].astype(str).isin(pred_filter)

    segment_val = SEGMENT_OPTIONS[segment_label]
    if segment_val == "high_confidence_wrong":
        mask &= df["high_confidence_wrong"]
    elif segment_val is not None:
        mask &= df["error_type"] == segment_val

    conf_lo, conf_hi = CONF_BUCKETS[conf_label]
    mask &= (df["break_confidence"] >= conf_lo) & (df["break_confidence"] < conf_hi)
    if status_filter is not None and status_col in df.columns:
        mask &= df[status_col].astype(str).isin(status_filter)
    if semantic_filter is not None and semantic_col in df.columns:
        mask &= df[semantic_col].astype(str).isin(semantic_filter)
    if fresh_filter is not None and fresh_col in df.columns:
        mask &= df[fresh_col].astype(str).isin(fresh_filter)
    if outcome_filter is not None and outcome_col in df.columns:
        mask &= df[outcome_col].astype(str).isin(outcome_filter)
    if dominant_filter is not None and dominant_col in df.columns:
        mask &= df[dominant_col].astype(str).isin(dominant_filter)
    if candidate_filter is not None:
        candidate_labels = np.where(df["candidate_gate"].astype(bool), "candidate", "not_candidate")
        mask &= pd.Series(candidate_labels, index=df.index).isin(candidate_filter)

    filtered = df[mask].reset_index(drop=True)

    if len(filtered) == 0:
        st.warning("No samples match the current filter.")
        st.stop()

    # ── Session state ─────────────────────────────────────────────────────────
    state_key = f"pos_{segment_label}_{conf_label}_{selected_horizon}_{display_past_bars}_{display_future_bars}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    pos = min(st.session_state[state_key], len(filtered) - 1)

    # ── Navigation ────────────────────────────────────────────────────────────
    with nav_slot:
        st.caption(f"{len(filtered)} matching samples")
        nav_prev, nav_next = st.columns(2)
        with nav_prev:
            if st.button("◀ Prev", use_container_width=True):
                st.session_state[state_key] = max(0, pos - 1)
                st.rerun()
        with nav_next:
            if st.button("Next ▶", use_container_width=True):
                st.session_state[state_key] = (
                    int(np.random.randint(0, len(filtered))) if random_mode else min(len(filtered) - 1, pos + 1)
                )
                st.rerun()
        nav_jump, nav_rand = st.columns([2, 1])
        with nav_jump:
            new_pos = st.number_input(
                "Sample",
                min_value=0,
                max_value=len(filtered) - 1,
                value=pos,
                step=1,
            )
            if new_pos != pos:
                st.session_state[state_key] = int(new_pos)
                st.rerun()
        with nav_rand:
            st.write("")
            st.write("")
            if st.button("🎲", use_container_width=True):
                st.session_state[state_key] = int(np.random.randint(0, len(filtered)))
                st.rerun()

    # ── Current row ───────────────────────────────────────────────────────────
    row = filtered.iloc[pos]
    symbol = str(row["symbol"])
    anchor_dt = row["anchor_dt"]

    ohlcv = load_ohlcv(symbol)
    window_df = get_window(
        ohlcv,
        anchor_dt,
        past_bars=display_past_bars,
        future_bars=display_future_bars,
    )

    if window_df is None:
        st.error(f"Could not load window for {symbol} @ {anchor_dt}. Check raw data coverage.")
        st.stop()

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = build_chart(
        window_df,
        row,
        selected_horizon=selected_horizon,
        past_bars=display_past_bars,
        future_bars=display_future_bars,
    )
    st.caption(
        f"{symbol} | {str(anchor_dt)[:16]} | true={row['true_label']} | pred={row['pred_label']} | "
        f"{row['error_type']} | {pos + 1}/{len(filtered)} | "
        f"plot={display_past_bars} past + {display_future_bars} future bars"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    col_meta, col_current, col_stage1b = st.columns([1.0, 1.2, 2.2], gap="small")

    with col_meta:
        st.subheader("Sample")
        st.metric("Symbol", symbol)
        st.metric("Timestamp", str(anchor_dt)[:16])
        st.metric("Position", f"{pos + 1} / {len(filtered)}")
        error_type = str(row.get("error_type", ""))
        error_colors_css = {
            "correct": "green",
            "false_positive_break": "orange",
            "false_negative_break": "#2196f3",
            "wrong_direction": "red",
        }
        color = error_colors_css.get(error_type, "gray")
        st.markdown(
            f"**Segment:** <span style='color:{color};font-weight:700'>"
            f"{error_type.replace('_', ' ')}</span>",
            unsafe_allow_html=True,
        )

    with col_current:
        st.subheader("Current structure")
        current_label = str(row.get("current_label", ""))
        color = LABEL_COLORS.get(current_label, "#ccc")
        st.markdown(
            f"**Stage 1A label:** <span style='color:{color};font-size:1.1em;font-weight:700'>"
            f"{LABEL_EMOJI.get(current_label, current_label)}</span>",
            unsafe_allow_html=True,
        )
        current_close = row.get("current_close")
        if pd.notna(current_close):
            st.metric("Current close", f"{current_close:.2f}")
        st.caption("Current anchor window and level proximity.")
        bull_dist = row.get("bull_distance_pct")
        bear_dist = row.get("bear_distance_pct")
        nearest = row.get("nearest_distance_pct")
        eff_break = row.get("effective_break_pct")
        prox_left, prox_right = st.columns(2)
        with prox_left:
            if pd.notna(bull_dist):
                st.metric("Bull dist", f"{bull_dist * 100:.3f}%")
            if pd.notna(nearest):
                st.metric("Nearest", f"{nearest * 100:.3f}%")
        with prox_right:
            if pd.notna(bear_dist):
                st.metric("Bear dist", f"{bear_dist * 100:.3f}%")
            if pd.notna(eff_break):
                st.metric("Break th", f"{eff_break * 100:.3f}%")

    with col_stage1b:
        st.subheader("H16 selected prediction")
        st.caption("Source: H16 joint event predictor CSV.")
        pred_left, pred_right = st.columns(2)
        pred_label = str(row.get("pred_label", ""))
        color = LABEL_COLORS.get(pred_label, "#ccc")
        true_label = str(row.get("true_label", ""))
        color2 = LABEL_COLORS.get(true_label, "#ccc")
        with pred_left:
            st.markdown(
                f"**Decision:** <span style='color:{color};font-weight:700'>"
                f"{LABEL_EMOJI.get(pred_label, pred_label)}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Target true:** <span style='color:{color2};font-weight:700'>"
                f"{LABEL_EMOJI.get(true_label, true_label)}</span>",
                unsafe_allow_html=True,
            )
            if "pred_joint_event" in row.index:
                st.metric("Pred joint", str(row["pred_joint_event"]))
                st.metric("True joint", str(row["true_joint_event"]))
        with pred_right:
            st.metric("Raw p_break", f"{row['p_break']:.3f}")
            st.metric("Decision conf", f"{row['break_confidence']:.3f}")
            if "pred_joint_event" in row.index:
                st.metric("Joint conf", f"{float(row['joint_confidence']):.3f}")
            if "candidate_strength" in row.index:
                st.metric("Candidate strength", f"{float(row['candidate_strength']):.3f}")
            if "candidate_gate" in row.index:
                gate = "candidate" if bool(row["candidate_gate"]) else "not candidate"
                st.metric("Candidate gate", gate)
            if pred_label == "none":
                st.caption("For `none`, confidence is `1 - p_break`.")

    def render_horizon_context(horizon: str) -> None:
        st.markdown(f"#### {horizon.upper()} label context")
        st.caption("Source: forward-label parquet. These are supervised labels/diagnostics, not model predictions.")
        c1, c2, c3 = st.columns(3)
        with c1:
            if f"{horizon}_break_anchor_status" in row.index:
                st.metric("Anchor status", str(row[f"{horizon}_break_anchor_status"]))
            if f"{horizon}_fresh_break_direction" in row.index:
                st.metric("Fresh direction", str(row[f"{horizon}_fresh_break_direction"]))
            if f"{horizon}_break_semantic_label" in row.index:
                st.metric("Semantic", str(row[f"{horizon}_break_semantic_label"]))
        with c2:
            if f"{horizon}_event_type" in row.index:
                st.metric("Event type", str(row[f"{horizon}_event_type"]))
            if f"{horizon}_event_direction" in row.index:
                st.metric("Event direction", str(row[f"{horizon}_event_direction"]))
            if f"{horizon}_post_break_outcome" in row.index:
                st.metric("Outcome", str(row[f"{horizon}_post_break_outcome"]))
            if f"{horizon}_dominant_forward_direction" in row.index:
                st.metric("Dominant", str(row[f"{horizon}_dominant_forward_direction"]))
        with c3:
            if f"{horizon}_time_to_break" in row.index:
                st.metric("Time to break", int(row[f"{horizon}_time_to_break"]))
            if f"{horizon}_bull_confirm_bar" in row.index:
                st.metric("Bull confirm", int(row[f"{horizon}_bull_confirm_bar"]))
            if f"{horizon}_bear_confirm_bar" in row.index:
                st.metric("Bear confirm", int(row[f"{horizon}_bear_confirm_bar"]))
            if f"{horizon}_bull_close_count" in row.index and f"{horizon}_bear_close_count" in row.index:
                st.metric(
                    "Close hits",
                    f"Bull {int(row[f'{horizon}_bull_close_count'])} / "
                    f"Bear {int(row[f'{horizon}_bear_close_count'])}",
                )

    st.divider()
    st.subheader("Forward label context")
    st.caption("This section explains the labels behind H8/H16. The selected model prediction is shown above.")
    h8_context, h16_context = st.columns(2, gap="medium")
    with h8_context:
        render_horizon_context("h8")
    with h16_context:
        render_horizon_context("h16")

    st.caption(
        f"Past zone = {display_past_bars}-bar display window. "
        f"Future zone = {display_future_bars}-bar display window. "
        f"Selected label horizon = {selected_horizon.upper()}. Vertical line = anchor bar."
    )


if __name__ == "__main__":
    main()
