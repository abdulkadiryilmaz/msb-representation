"""Stage 2 actionability inspector: browse labels with price path context.

Usage:
    streamlit run scripts/inspect_stage2_actionability.py
    streamlit run scripts/inspect_stage2_actionability.py -- --labels path/to/stage2_actionability_labels.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PAST_BARS = 48
FUTURE_BARS = 16
H8_BARS = 8

RAW_DATA_ROOT = Path("data/raw/binance")
DEFAULT_LABELS = "data/stage2/binance/15m/stage2_actionability_v1_selected_test/stage2_actionability_labels.csv"

LABEL_COLORS = {
    "no_trade": "#9e9e9e",
    "wait_for_break": "#42a5f5",
    "wait_for_retest": "#ffb74d",
    "actionable_break_candidate": "#26a69a",
}

CONF_BUCKETS = {
    "All confidence": (0.0, 1.01),
    "Low (<0.70)": (0.0, 0.70),
    "Medium (0.70-0.90)": (0.70, 0.90),
    "High (0.90-1.00)": (0.90, 1.01),
}


@st.cache_data
def load_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["anchor_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_localize(None)
    return df


@st.cache_data
def load_ohlcv(symbol: str) -> pd.DataFrame:
    path = RAW_DATA_ROOT / f"{symbol}.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


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


def _hex_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


def _maybe_hline(fig: go.Figure, value: object, *, label: str, color: str, dash: str = "dash") -> None:
    if pd.isna(value):
        return
    y = float(value)
    fig.add_trace(
        go.Scatter(
            x=[-0.5, PAST_BARS + FUTURE_BARS - 0.5],
            y=[y, y],
            mode="lines",
            name=f"{label} {y:.4f}",
            line=dict(color=color, width=1.4, dash=dash),
            hovertemplate=f"{label}: {y:.4f}<extra></extra>",
        )
    )


def _maybe_event_line(
    fig: go.Figure,
    value: object,
    *,
    label: str,
    color: str,
    y_paper: float,
    dash: str = "dot",
) -> None:
    if pd.isna(value):
        return
    offset = int(value)
    if offset < 1:
        return
    x = PAST_BARS - 1 + offset
    fig.add_shape(
        type="line",
        x0=x,
        x1=x,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color=color, width=1.2, dash=dash),
    )
    fig.add_annotation(
        x=x,
        y=y_paper,
        xref="x",
        yref="paper",
        text=label,
        showarrow=False,
        font=dict(size=10, color=color),
        bgcolor="rgba(15,15,35,0.78)",
        bordercolor=color,
        borderwidth=1,
        borderpad=2,
    )


def build_chart(window_df: pd.DataFrame, row: pd.Series) -> go.Figure:
    xs = list(range(len(window_df)))
    anchor_x = PAST_BARS - 1
    future_start = PAST_BARS
    h8_end = PAST_BARS + H8_BARS - 0.5
    h16_end = PAST_BARS + FUTURE_BARS - 0.5
    label_name = str(row.get("label_name", ""))
    label_color = LABEL_COLORS.get(label_name, "#9e9e9e")

    fig = go.Figure()
    fig.add_vrect(
        x0=-0.5,
        x1=anchor_x + 0.5,
        fillcolor="rgba(100,100,100,0.07)",
        line_width=0,
    )
    fig.add_vrect(
        x0=future_start - 0.5,
        x1=h8_end,
        fillcolor=f"rgba({_hex_rgb(label_color)},0.15)",
        line_width=0,
    )
    fig.add_vrect(
        x0=h8_end,
        x1=h16_end,
        fillcolor="rgba(66,165,245,0.08)",
        line_width=0,
    )

    fig.add_trace(
        go.Candlestick(
            x=xs,
            open=window_df["open"].tolist(),
            high=window_df["high"].tolist(),
            low=window_df["low"].tolist(),
            close=window_df["close"].tolist(),
            name="price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
        )
    )
    fig.add_vline(x=anchor_x + 0.5, line_dash="dash", line_color="#aaaaaa", line_width=1.5)

    _maybe_hline(fig, row.get("trigger_level"), label="trigger", color="#42a5f5")
    _maybe_hline(fig, row.get("invalidation_level"), label="invalidation", color="#ef5350")
    _maybe_hline(fig, row.get("target_level"), label="target", color="#26a69a")
    _maybe_hline(fig, row.get("bull_level"), label="bull", color="#80cbc4", dash="dot")
    _maybe_hline(fig, row.get("bear_level"), label="bear", color="#ef9a9a", dash="dot")

    _maybe_event_line(fig, row.get("break_time_bars"), label="break", color="#42a5f5", y_paper=1.00)
    _maybe_event_line(fig, row.get("first_target_time"), label="target", color="#26a69a", y_paper=0.94)
    _maybe_event_line(fig, row.get("first_invalidation_time"), label="invalid", color="#ef5350", y_paper=0.88)
    _maybe_event_line(fig, row.get("first_retest_time"), label="retest", color="#ffb74d", y_paper=0.82)

    fig.add_annotation(
        x=PAST_BARS / 2,
        y=1.02,
        xref="x",
        yref="paper",
        text="past 48",
        showarrow=False,
        font=dict(size=10, color="#777"),
    )
    fig.add_annotation(
        x=PAST_BARS + H8_BARS / 2,
        y=1.02,
        xref="x",
        yref="paper",
        text="H8",
        showarrow=False,
        font=dict(size=10, color=label_color),
    )
    fig.add_annotation(
        x=PAST_BARS + H8_BARS + (FUTURE_BARS - H8_BARS) / 2,
        y=1.02,
        xref="x",
        yref="paper",
        text="H16",
        showarrow=False,
        font=dict(size=10, color="#42a5f5"),
    )

    title = label_name.replace("_", " ").upper()
    reason = str(row.get("reason_code", ""))
    fig.update_layout(
        title=dict(
            text=f"<b style='color:{label_color}'>{title}</b> <span style='font-size:13px;color:#aaa'>{reason}</span>",
            font=dict(size=18),
            x=0.5,
        ),
        xaxis_title="bar index",
        yaxis_title="price",
        xaxis_rangeslider_visible=False,
        height=540,
        margin=dict(l=40, r=32, t=82, b=40),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f23",
        font=dict(color="#cccccc"),
        xaxis=dict(gridcolor="#2a2a3e", tickfont=dict(color="#888")),
        yaxis=dict(gridcolor="#2a2a3e", tickfont=dict(color="#888")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
    )
    return fig


def _fmt_pct(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.3f}%"


def _fmt_float(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def main() -> None:
    st.set_page_config(page_title="Stage 2 Inspector", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
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
        [data-testid="stMetricLabel"] {
            font-size: 0.68rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 0.92rem;
        }
        div[data-testid="stVerticalBlock"] {
            gap: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("Stage 2\nActionability")
        st.caption("H8 decision + H16 path sanity")
        st.divider()
        st.header("Navigation")
        nav_slot = st.container()
        st.header("Data")
        cli_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
        default_csv = DEFAULT_LABELS
        if "--labels" in cli_args:
            default_csv = cli_args[cli_args.index("--labels") + 1]
        csv_path = st.text_input("Stage 2 labels CSV", value=default_csv)
        if not Path(csv_path).exists():
            st.error(f"File not found: {csv_path}")
            st.stop()
        df = load_labels(csv_path)

        st.divider()
        st.header("Filters")
        symbols = sorted(df["symbol"].unique())
        labels = sorted(df["label_name"].unique())
        reasons = sorted(df["reason_code"].unique())
        symbol_filter = st.multiselect("Symbol", options=symbols, default=symbols)
        label_filter = st.multiselect(
            "Actionability",
            options=labels,
            default=[label for label in ["actionable_break_candidate", "wait_for_retest"] if label in labels],
        )
        reason_filter = st.multiselect("Reason", options=reasons, default=reasons)
        conf_label = st.selectbox("Confidence", options=list(CONF_BUCKETS.keys()))
        stage1b_errors = sorted(df["stage1b_error_type"].dropna().unique())
        error_filter = st.multiselect("Stage 1B error", options=stage1b_errors, default=stage1b_errors)
        random_mode = st.checkbox("Random jump", value=False)

    mask = df["symbol"].isin(symbol_filter)
    mask &= df["label_name"].isin(label_filter)
    mask &= df["reason_code"].isin(reason_filter)
    mask &= df["stage1b_error_type"].isin(error_filter)
    conf_lo, conf_hi = CONF_BUCKETS[conf_label]
    mask &= (df["break_confidence"] >= conf_lo) & (df["break_confidence"] < conf_hi)
    filtered = df[mask].reset_index(drop=True)

    if len(filtered) == 0:
        st.warning("No samples match the current filter.")
        st.stop()

    state_key = f"stage2_pos_{','.join(label_filter)}_{conf_label}_{','.join(error_filter)}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    pos = min(st.session_state[state_key], len(filtered) - 1)

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

    row = filtered.iloc[pos]
    symbol = str(row["symbol"])
    anchor_dt = row["anchor_dt"]
    window_df = get_window(load_ohlcv(symbol), anchor_dt)
    if window_df is None:
        st.error(f"Could not load window for {symbol} @ {anchor_dt}.")
        st.stop()

    st.caption(
        f"{symbol} | {str(anchor_dt)[:16]} | {row['label_name']} | "
        f"{row['reason_code']} | {pos + 1}/{len(filtered)}"
    )
    st.plotly_chart(build_chart(window_df, row), use_container_width=True)

    col_sample, col_stage2, col_stage1b, col_path, col_levels = st.columns(5, gap="small")
    with col_sample:
        st.subheader("Sample")
        st.metric("Symbol", symbol)
        st.metric("Timestamp", str(anchor_dt)[:16])
        st.metric("Position", f"{pos + 1} / {len(filtered)}")
        st.metric("Index", int(row["index"]))
    with col_stage2:
        st.subheader("Stage 2")
        st.metric("Label", str(row["label_name"]))
        st.metric("Reason", str(row["reason_code"]))
        st.metric("Side", str(row["side"]))
        st.metric("Distance", str(row["distance_bucket"]))
    with col_stage1b:
        st.subheader("Stage 1B")
        st.metric("Pred", str(row["stage1b_pred_label"]))
        st.metric("True", str(row["stage1b_true_label"]))
        st.metric("Error", str(row["stage1b_error_type"]))
        st.metric("Break conf", _fmt_float(row["break_confidence"]))
    with col_path:
        st.subheader("Path")
        st.metric("MFE R", _fmt_float(row["mfe_r"]))
        st.metric("MAE R", _fmt_float(row["mae_r"]))
        st.metric("Break bar", "-" if pd.isna(row["break_time_bars"]) else int(row["break_time_bars"]))
        st.metric("Retest bar", "-" if pd.isna(row["first_retest_time"]) else int(row["first_retest_time"]))
    with col_levels:
        st.subheader("Levels")
        st.metric("Trigger dist", _fmt_pct(row["distance_to_trigger"]))
        st.metric("Risk", _fmt_float(row["risk"]))
        st.metric("Trigger", _fmt_float(row["trigger_level"]))
        st.metric("Target", _fmt_float(row["target_level"]))

    st.caption("Past = 48 bars. H8 is the decision horizon. H16 is the observation horizon for retest/path sanity.")


if __name__ == "__main__":
    main()
