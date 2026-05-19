"""Stage 1B predictor inspector: browse H8 predictions alongside candlestick charts.

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

PAST_BARS = 48
FUTURE_BARS = 8

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
    "stage1b_h8_predictor_v1a_z_fused/proximity_audit_test/predictions_with_proximity.csv"
)
RAW_DATA_ROOT = Path("data/raw/binance")


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


def get_window(ohlcv: pd.DataFrame, anchor_dt: pd.Timestamp) -> pd.DataFrame | None:
    matches = ohlcv.index[ohlcv["timestamp"] == anchor_dt].tolist()
    if not matches:
        return None
    anchor_idx = matches[0]
    start = anchor_idx - PAST_BARS + 1
    end = anchor_idx + FUTURE_BARS + 1
    if start < 0 or end > len(ohlcv):
        return None
    return ohlcv.iloc[start:end].reset_index(drop=True)


# ── Chart ─────────────────────────────────────────────────────────────────────

def build_chart(window_df: pd.DataFrame, row: pd.Series) -> go.Figure:
    opens  = window_df["open"].tolist()
    highs  = window_df["high"].tolist()
    lows   = window_df["low"].tolist()
    closes = window_df["close"].tolist()
    n = len(closes)
    xs = list(range(n))

    anchor_x = PAST_BARS - 1  # bar index 47
    future_start = PAST_BARS   # bar index 48

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
        annotation_text="past (48 bars)", annotation_position="top left",
        annotation_font_size=10, annotation_font_color="#666",
    )
    fig.add_vrect(
        x0=future_start - 0.5, x1=n - 0.5,
        fillcolor=f"rgba({hex_rgb(future_color_hex)},0.12)", line_width=0,
        annotation_text="H8 future", annotation_position="top right",
        annotation_font_size=10, annotation_font_color=future_color_hex,
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
    title_text = error_type.replace("_", " ").upper()

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
    )
    st.title("Stage 1B — Predictor Inspector")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
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

        st.divider()
        st.header("Navigation")
        random_mode = st.checkbox("Random jump", value=False)

    # ── Filter ────────────────────────────────────────────────────────────────
    mask = df["symbol"].isin(symbol_filter)

    segment_val = SEGMENT_OPTIONS[segment_label]
    if segment_val == "high_confidence_wrong":
        mask &= df["high_confidence_wrong"]
    elif segment_val is not None:
        mask &= df["error_type"] == segment_val

    conf_lo, conf_hi = CONF_BUCKETS[conf_label]
    mask &= (df["break_confidence"] >= conf_lo) & (df["break_confidence"] < conf_hi)

    filtered = df[mask].reset_index(drop=True)

    if len(filtered) == 0:
        st.warning("No samples match the current filter.")
        st.stop()

    # ── Session state ─────────────────────────────────────────────────────────
    state_key = f"pos_{segment_label}_{conf_label}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    pos = min(st.session_state[state_key], len(filtered) - 1)

    # ── Navigation ────────────────────────────────────────────────────────────
    col_prev, col_idx, col_next, col_rand = st.columns([1, 3, 1, 1])

    with col_prev:
        if st.button("◀ Prev", use_container_width=True):
            st.session_state[state_key] = max(0, pos - 1)
            st.rerun()
    with col_next:
        if st.button("Next ▶", use_container_width=True):
            if random_mode:
                st.session_state[state_key] = int(np.random.randint(0, len(filtered)))
            else:
                st.session_state[state_key] = min(len(filtered) - 1, pos + 1)
            st.rerun()
    with col_rand:
        if st.button("🎲 Random", use_container_width=True):
            st.session_state[state_key] = int(np.random.randint(0, len(filtered)))
            st.rerun()
    with col_idx:
        new_pos = st.number_input(
            f"Sample (0 – {len(filtered) - 1})",
            min_value=0, max_value=len(filtered) - 1,
            value=pos, step=1, label_visibility="collapsed",
        )
        if new_pos != pos:
            st.session_state[state_key] = int(new_pos)
            st.rerun()

    # ── Current row ───────────────────────────────────────────────────────────
    row = filtered.iloc[pos]
    symbol = str(row["symbol"])
    anchor_dt = row["anchor_dt"]

    ohlcv = load_ohlcv(symbol)
    window_df = get_window(ohlcv, anchor_dt)

    if window_df is None:
        st.error(f"Could not load window for {symbol} @ {anchor_dt}. Check raw data coverage.")
        st.stop()

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = build_chart(window_df, row)
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    col_meta, col_stage1a, col_stage1b, col_proximity = st.columns(4)

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

    with col_stage1a:
        st.subheader("Stage 1A")
        s1a_label = str(row.get("true_label", ""))
        color = LABEL_COLORS.get(s1a_label, "#ccc")
        st.markdown(
            f"**Label:** <span style='color:{color};font-size:1.1em;font-weight:700'>"
            f"{LABEL_EMOJI.get(s1a_label, s1a_label)}</span>",
            unsafe_allow_html=True,
        )

    with col_stage1b:
        st.subheader("Stage 1B prediction")
        pred_label = str(row.get("pred_label", ""))
        color = LABEL_COLORS.get(pred_label, "#ccc")
        st.markdown(
            f"**Predicted:** <span style='color:{color};font-weight:700'>"
            f"{LABEL_EMOJI.get(pred_label, pred_label)}</span>",
            unsafe_allow_html=True,
        )
        true_label = str(row.get("true_label", ""))
        color2 = LABEL_COLORS.get(true_label, "#ccc")
        st.markdown(
            f"**True:** <span style='color:{color2};font-weight:700'>"
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
        st.metric("p_break", f"{row['p_break']:.3f}")
        st.metric("break confidence", f"{row['break_confidence']:.3f}")

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

    st.caption("Past zone = 48-bar short window. Future zone = H8 (8 bars). Vertical line = anchor bar.")


if __name__ == "__main__":
    main()
