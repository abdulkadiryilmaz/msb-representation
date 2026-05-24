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
    "stage1b_h8_fresh_break_v3_z_fused_proximity/proximity_audit_test_t057/predictions_with_proximity.csv"
)
RAW_DATA_ROOT = Path("data/raw/binance")
HORIZON_OPTIONS = ["h4", "h8", "h16", "h32", "h48"]


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_predictions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["anchor_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_localize(None)
    df["high_confidence_wrong"] = (~df["is_correct"]) & (df["break_confidence"] >= 0.90)
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
    future_color_hex = LABEL_COLORS.get(true_label, "#9e9e9e")

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
    }
    title_color = error_colors.get(error_type, "#cccccc")
    semantic = str(row.get(f"{selected_horizon}_break_semantic_label", ""))
    outcome = str(row.get(f"{selected_horizon}_post_break_outcome", ""))
    title_text = error_type.replace("_", " ").upper()
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

        csv_path = st.text_input("Predictions CSV", value=default_csv)
        if not Path(csv_path).exists():
            st.error(f"File not found: {csv_path}")
            st.stop()

        df = load_predictions(csv_path)

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
    col_meta, col_current, col_target, col_stage1b, col_proximity = st.columns(5, gap="small")

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
        st.caption("This is the structure label for the current anchor window.")

    with col_target:
        st.subheader("Forward target")
        target_label = str(row.get("true_label", ""))
        color = LABEL_COLORS.get(target_label, "#ccc")
        st.markdown(
            f"**Target true:** <span style='color:{color};font-size:1.1em;font-weight:700'>"
            f"{LABEL_EMOJI.get(target_label, target_label)}</span>",
            unsafe_allow_html=True,
        )
        st.caption("This is the supervised future target used by the predictor CSV.")

    with col_stage1b:
        st.subheader("Predictor decision")
        pred_label = str(row.get("pred_label", ""))
        color = LABEL_COLORS.get(pred_label, "#ccc")
        st.markdown(
            f"**Decision:** <span style='color:{color};font-weight:700'>"
            f"{LABEL_EMOJI.get(pred_label, pred_label)}</span>",
            unsafe_allow_html=True,
        )
        true_label = str(row.get("true_label", ""))
        color2 = LABEL_COLORS.get(true_label, "#ccc")
        st.markdown(
            f"**Target true:** <span style='color:{color2};font-weight:700'>"
            f"{LABEL_EMOJI.get(true_label, true_label)}</span>",
            unsafe_allow_html=True,
        )
        pred_dir = str(row.get("pred_direction", ""))
        color3 = LABEL_COLORS.get(pred_dir, "#ccc")
        st.markdown(
            f"**Direction head:** <span style='color:{color3};font-weight:700'>"
            f"{LABEL_EMOJI.get(pred_dir, pred_dir)}</span>"
            f" <span style='color:#888;font-size:0.85em'>({row['direction_confidence']:.2f})</span>",
            unsafe_allow_html=True,
        )
        st.metric("Raw p_break", f"{row['p_break']:.3f}")
        st.metric("Decision confidence", f"{row['break_confidence']:.3f}")
        if pred_label == "none":
            st.caption("For a `none` decision, confidence is `1 - p_break`.")

    with col_proximity:
        st.subheader("Proximity")
        bull_dist = row.get("bull_distance_pct")
        bear_dist = row.get("bear_distance_pct")
        nearest = row.get("nearest_distance_pct")
        eff_break = row.get("effective_break_pct")

        if pd.notna(bull_dist):
            st.metric("Bull distance", f"{bull_dist * 100:.3f}%")
        if pd.notna(bear_dist):
            st.metric("Bear distance", f"{bear_dist * 100:.3f}%")
        if pd.notna(nearest):
            st.metric("Nearest distance", f"{nearest * 100:.3f}%")
        if pd.notna(eff_break):
            st.metric("Break threshold", f"{eff_break * 100:.3f}%")

    v2_cols = [
        f"{selected_horizon}_break_anchor_status",
        f"{selected_horizon}_fresh_break_direction",
        f"{selected_horizon}_break_semantic_label",
        f"{selected_horizon}_post_break_outcome",
        f"{selected_horizon}_dominant_forward_direction",
        f"{selected_horizon}_event_type",
        f"{selected_horizon}_event_direction",
        f"{selected_horizon}_time_to_break",
        f"{selected_horizon}_bull_confirm_bar",
        f"{selected_horizon}_bear_confirm_bar",
        f"{selected_horizon}_bull_close_count",
        f"{selected_horizon}_bear_close_count",
    ]
    if any(col in row.index for col in v2_cols):
        st.subheader(f"Event-sequence context · {selected_horizon.upper()}")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if f"{selected_horizon}_break_anchor_status" in row.index:
                st.metric("Anchor status", str(row[f"{selected_horizon}_break_anchor_status"]))
            if "current_close" in row.index and pd.notna(row["current_close"]):
                st.metric("Current close", f"{float(row['current_close']):.4f}")
        with c2:
            if f"{selected_horizon}_fresh_break_direction" in row.index:
                st.metric("Fresh direction", str(row[f"{selected_horizon}_fresh_break_direction"]))
            if f"{selected_horizon}_break_semantic_label" in row.index:
                st.metric("Semantic", str(row[f"{selected_horizon}_break_semantic_label"]))
        with c3:
            if f"{selected_horizon}_event_type" in row.index:
                st.metric("Event type", str(row[f"{selected_horizon}_event_type"]))
            if f"{selected_horizon}_event_direction" in row.index:
                st.metric("Event direction", str(row[f"{selected_horizon}_event_direction"]))
            if f"{selected_horizon}_post_break_outcome" in row.index:
                st.metric("Outcome", str(row[f"{selected_horizon}_post_break_outcome"]))
            if f"{selected_horizon}_dominant_forward_direction" in row.index:
                st.metric("Dominant", str(row[f"{selected_horizon}_dominant_forward_direction"]))
        with c4:
            if f"{selected_horizon}_time_to_break" in row.index:
                st.metric("Time to break", int(row[f"{selected_horizon}_time_to_break"]))
            if f"{selected_horizon}_bull_confirm_bar" in row.index:
                st.metric("Bull confirm", int(row[f"{selected_horizon}_bull_confirm_bar"]))
            if f"{selected_horizon}_bear_confirm_bar" in row.index:
                st.metric("Bear confirm", int(row[f"{selected_horizon}_bear_confirm_bar"]))
            if f"{selected_horizon}_bull_close_count" in row.index and f"{selected_horizon}_bear_close_count" in row.index:
                st.metric(
                    "Close hits",
                    f"Bull {int(row[f'{selected_horizon}_bull_close_count'])} / "
                    f"Bear {int(row[f'{selected_horizon}_bear_close_count'])}",
                )

    st.caption(
        f"Past zone = {display_past_bars}-bar display window. "
        f"Future zone = {display_future_bars}-bar display window. "
        f"Selected label horizon = {selected_horizon.upper()}. Vertical line = anchor bar."
    )


if __name__ == "__main__":
    main()
