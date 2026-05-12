"""Label inspector: interactive candlestick visualization of Stage 1A labels.

Usage:
    streamlit run scripts/inspect_labels.py
    streamlit run scripts/inspect_labels.py -- --data-root data/stage1a/binance/15m
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from msb_repr.stage1a.config import Stage1ADatasetSpec
from msb_repr.stage1a.features import STAGE1A_SHORT_FEATURE_COLUMNS

# ── Constants ────────────────────────────────────────────────────────────────

LABEL_NAMES = {0: "intact", 1: "bullish", 2: "bearish"}
LABEL_COLORS = {0: "#9e9e9e", 1: "#26a69a", 2: "#ef5350"}
LABEL_EMOJI = {0: "⬜ intact", 1: "🟢 bullish", 2: "🔴 bearish"}

OPEN_IDX = STAGE1A_SHORT_FEATURE_COLUMNS.index("open")
HIGH_IDX = STAGE1A_SHORT_FEATURE_COLUMNS.index("high")
LOW_IDX = STAGE1A_SHORT_FEATURE_COLUMNS.index("low")
CLOSE_IDX = STAGE1A_SHORT_FEATURE_COLUMNS.index("close")


# ── Label diagnostics ────────────────────────────────────────────────────────

def _find_last_pivot_high_with_idx(highs: np.ndarray, n: int) -> tuple[float, int] | tuple[None, None]:
    for i in range(len(highs) - n - 1, n - 1, -1):
        val = highs[i]
        if val > highs[i - n:i].max() and val > highs[i + 1:i + n + 1].max():
            return float(val), i
    return None, None


def _find_last_pivot_low_with_idx(lows: np.ndarray, n: int) -> tuple[float, int] | tuple[None, None]:
    for i in range(len(lows) - n - 1, n - 1, -1):
        val = lows[i]
        if val < lows[i - n:i].min() and val < lows[i + 1:i + n + 1].min():
            return float(val), i
    return None, None


def compute_diagnostics(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, cfg) -> dict:
    structure_bars = cfg.structure_bars
    recent_bars = cfg.recent_bars

    struct_highs = highs[:structure_bars]
    struct_lows = lows[:structure_bars]
    recent_closes = closes[structure_bars:structure_bars + recent_bars]

    atr = float(np.mean(struct_highs - struct_lows))
    ref_close = float(closes[structure_bars - 1]) if closes[structure_bars - 1] > 0 else 1.0
    effective_break_pct = max(cfg.min_break_pct, cfg.atr_factor * atr / ref_close)

    swing_high, pivot_high_idx = _find_last_pivot_high_with_idx(struct_highs, cfg.pivot_n)
    swing_low, pivot_low_idx = _find_last_pivot_low_with_idx(struct_lows, cfg.pivot_n)

    bull_level = swing_high * (1.0 + effective_break_pct) if swing_high is not None else None
    bear_level = swing_low * (1.0 - effective_break_pct) if swing_low is not None else None

    bull_bars_above = int(np.sum(recent_closes > bull_level)) if bull_level is not None else 0
    bear_bars_below = int(np.sum(recent_closes < bear_level)) if bear_level is not None else 0

    bull_strength = -np.inf
    bear_strength = -np.inf
    if bull_level is not None:
        excess = recent_closes - bull_level
        if bull_bars_above >= cfg.min_recent_break_bars:
            bull_strength = float(np.max(excess / max(bull_level, 1e-8)))
    if bear_level is not None:
        excess = bear_level - recent_closes
        if bear_bars_below >= cfg.min_recent_break_bars:
            bear_strength = float(np.max(excess / max(bear_level, 1e-8)))

    return {
        "atr": atr,
        "effective_break_pct": effective_break_pct,
        "swing_high": swing_high,
        "pivot_high_idx": pivot_high_idx,
        "swing_low": swing_low,
        "pivot_low_idx": pivot_low_idx,
        "bull_level": bull_level,
        "bear_level": bear_level,
        "bull_bars_above": bull_bars_above,
        "bear_bars_below": bear_bars_below,
        "bull_strength": bull_strength,
        "bear_strength": bear_strength,
        "min_recent_break_bars": cfg.min_recent_break_bars,
    }


# ── Chart builder ─────────────────────────────────────────────────────────────

def build_chart(window: np.ndarray, diag: dict, label: int, cfg) -> go.Figure:
    opens = window[OPEN_IDX]
    highs = window[HIGH_IDX]
    lows = window[LOW_IDX]
    closes = window[CLOSE_IDX]
    n_bars = len(closes)
    xs = list(range(n_bars))

    structure_end = cfg.structure_bars - 1  # last bar of structure zone (inclusive)
    recent_start = cfg.structure_bars       # first bar of recent zone

    fig = go.Figure()

    # Zone backgrounds
    fig.add_vrect(
        x0=-0.5, x1=structure_end + 0.5,
        fillcolor="rgba(100,100,100,0.08)",
        line_width=0,
        annotation_text="structure zone",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color="#888",
    )
    fig.add_vrect(
        x0=recent_start - 0.5, x1=n_bars - 0.5,
        fillcolor=f"rgba({_hex_to_rgb(LABEL_COLORS[label])},0.10)",
        line_width=0,
        annotation_text="recent zone",
        annotation_position="top right",
        annotation_font_size=11,
        annotation_font_color=LABEL_COLORS[label],
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=xs,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        name="price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
    ))

    # Structure / recent zone divider
    fig.add_vline(
        x=recent_start - 0.5,
        line_dash="dash",
        line_color="#aaaaaa",
        line_width=1,
    )

    price_range = float(np.max(highs) - np.min(lows))
    marker_offset = price_range * 0.012

    # Swing high pivot marker + levels
    if diag["swing_high"] is not None:
        pivot_x = diag["pivot_high_idx"]
        sh = diag["swing_high"]
        bl = diag["bull_level"]

        # Pivot marker
        fig.add_trace(go.Scatter(
            x=[pivot_x], y=[sh + marker_offset * 2],
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="#26a69a"),
            name="swing high",
            showlegend=True,
        ))

        # Swing high horizontal line (structure zone only)
        fig.add_shape(type="line", x0=pivot_x, x1=structure_end,
                      y0=sh, y1=sh,
                      line=dict(color="#26a69a", width=1, dash="dot"))

        # Bull break level (full width)
        fig.add_hline(
            y=bl,
            line_dash="dash",
            line_color="#26a69a",
            line_width=1.5,
            annotation_text=f"bull level {bl:.4f}",
            annotation_position="right",
            annotation_font_size=10,
            annotation_font_color="#26a69a",
        )

    # Swing low pivot marker + levels
    if diag["swing_low"] is not None:
        pivot_x = diag["pivot_low_idx"]
        sl = diag["swing_low"]
        bl = diag["bear_level"]

        # Pivot marker
        fig.add_trace(go.Scatter(
            x=[pivot_x], y=[sl - marker_offset * 2],
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="#ef5350"),
            name="swing low",
            showlegend=True,
        ))

        # Swing low horizontal line (structure zone only)
        fig.add_shape(type="line", x0=pivot_x, x1=structure_end,
                      y0=sl, y1=sl,
                      line=dict(color="#ef5350", width=1, dash="dot"))

        # Bear break level (full width)
        fig.add_hline(
            y=bl,
            line_dash="dash",
            line_color="#ef5350",
            line_width=1.5,
            annotation_text=f"bear level {bl:.4f}",
            annotation_position="right",
            annotation_font_size=10,
            annotation_font_color="#ef5350",
        )

    label_color = LABEL_COLORS[label]
    label_name = LABEL_NAMES[label].upper()
    fig.update_layout(
        title=dict(
            text=f"<b style='color:{label_color}'>{label_name}</b>",
            font=dict(size=20),
            x=0.5,
        ),
        xaxis_title="bar index",
        yaxis_title="price",
        xaxis_rangeslider_visible=False,
        height=480,
        margin=dict(l=40, r=120, t=60, b=40),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0f0f23",
        font=dict(color="#cccccc"),
        xaxis=dict(gridcolor="#2a2a3e", tickfont=dict(color="#888")),
        yaxis=dict(gridcolor="#2a2a3e", tickfont=dict(color="#888")),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11),
        ),
    )
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_symbol_data(npz_path: str) -> dict:
    data = np.load(npz_path)
    return {
        "short_windows": data["short_windows"],
        "labels": data["labels"],
        "timestamps": data["timestamps"],
    }


@st.cache_data
def load_spec(spec_path: str) -> Stage1ADatasetSpec:
    return Stage1ADatasetSpec.load(Path(spec_path))


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Label Inspector",
        page_icon="🔍",
        layout="wide",
    )
    st.title("Stage 1A — Label Inspector")

    # ── Sidebar: data settings ────────────────────────────────────────────────
    with st.sidebar:
        st.header("Dataset")

        # CLI --data-root override or sidebar input
        cli_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
        default_root = "data/stage1a/binance/15m"
        if "--data-root" in cli_args:
            default_root = cli_args[cli_args.index("--data-root") + 1]

        data_root = st.text_input("Data root", value=default_root)
        spec_path = Path(data_root) / "spec.json"

        if not spec_path.exists():
            st.error(f"spec.json not found at {spec_path}")
            st.stop()

        spec = load_spec(str(spec_path))
        symbol = st.selectbox("Symbol", options=spec.symbols)

        st.divider()
        st.header("Filters")

        label_filter = st.multiselect(
            "Label",
            options=[0, 1, 2],
            default=[0, 1, 2],
            format_func=lambda x: LABEL_EMOJI[x],
        )

        st.divider()
        st.header("Navigation")

        random_mode = st.checkbox("Random jump", value=False)

    # ── Load data ─────────────────────────────────────────────────────────────
    npz_path = Path(data_root) / "symbols" / f"{symbol}.npz"
    if not npz_path.exists():
        st.error(f"NPZ not found: {npz_path}")
        st.stop()

    data = load_symbol_data(str(npz_path))
    all_labels = data["labels"]
    all_windows = data["short_windows"]

    # Build filtered index
    filtered_idx = np.where(np.isin(all_labels, label_filter))[0]
    if len(filtered_idx) == 0:
        st.warning("No samples match the current filter.")
        st.stop()

    # ── Session state for current position ───────────────────────────────────
    state_key = f"pos_{symbol}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    pos = st.session_state[state_key]
    pos = min(pos, len(filtered_idx) - 1)

    # ── Navigation controls ───────────────────────────────────────────────────
    col_prev, col_idx, col_next, col_rand = st.columns([1, 3, 1, 1])

    with col_prev:
        if st.button("◀ Prev", use_container_width=True):
            st.session_state[state_key] = max(0, pos - 1)
            st.rerun()

    with col_next:
        if st.button("Next ▶", use_container_width=True):
            if random_mode:
                st.session_state[state_key] = int(np.random.randint(0, len(filtered_idx)))
            else:
                st.session_state[state_key] = min(len(filtered_idx) - 1, pos + 1)
            st.rerun()

    with col_rand:
        if st.button("🎲 Random", use_container_width=True):
            st.session_state[state_key] = int(np.random.randint(0, len(filtered_idx)))
            st.rerun()

    with col_idx:
        new_pos = st.number_input(
            f"Sample (0 – {len(filtered_idx) - 1})",
            min_value=0,
            max_value=len(filtered_idx) - 1,
            value=pos,
            step=1,
            label_visibility="collapsed",
        )
        if new_pos != pos:
            st.session_state[state_key] = int(new_pos)
            st.rerun()

    # ── Current sample ────────────────────────────────────────────────────────
    sample_idx = int(filtered_idx[pos])
    window = all_windows[sample_idx]  # (10, 48)
    label = int(all_labels[sample_idx])

    highs = window[HIGH_IDX]
    lows = window[LOW_IDX]
    closes = window[CLOSE_IDX]

    diag = compute_diagnostics(highs, lows, closes, spec.labels)

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = build_chart(window, diag, label, spec.labels)
    st.plotly_chart(fig, use_container_width=True)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    col_meta, col_struct, col_break = st.columns(3)

    with col_meta:
        st.subheader("Sample info")
        st.metric("Symbol", symbol)
        st.metric("Global index", sample_idx)
        st.metric("Filtered position", f"{pos + 1} / {len(filtered_idx)}")
        label_color = LABEL_COLORS[label]
        st.markdown(
            f"**Label:** <span style='color:{label_color};font-size:1.2em;font-weight:700'>"
            f"{LABEL_EMOJI[label]}</span>",
            unsafe_allow_html=True,
        )

    with col_struct:
        st.subheader("Structure zone")
        st.metric("ATR (mean HL)", f"{diag['atr']:.4f}")
        st.metric("Break threshold", f"{diag['effective_break_pct'] * 100:.3f}%")
        if diag["swing_high"] is not None:
            st.metric("Swing high (bar)", f"{diag['pivot_high_idx']}")
            st.metric("Swing high value", f"{diag['swing_high']:.4f}")
        else:
            st.metric("Swing high", "not found")
        if diag["swing_low"] is not None:
            st.metric("Swing low (bar)", f"{diag['pivot_low_idx']}")
            st.metric("Swing low value", f"{diag['swing_low']:.4f}")
        else:
            st.metric("Swing low", "not found")

    with col_break:
        st.subheader("Recent zone")
        min_bars = diag["min_recent_break_bars"]
        if diag["bull_level"] is not None:
            bull_ok = diag["bull_bars_above"] >= min_bars
            st.metric(
                f"Bull level ({diag['bull_bars_above']}/{min_bars} bars above)",
                f"{diag['bull_level']:.4f}",
                delta="TRIGGERED" if bull_ok else "not triggered",
                delta_color="normal" if bull_ok else "off",
            )
            if diag["bull_strength"] > -np.inf:
                st.metric("Bull strength", f"{diag['bull_strength']:.5f}")
        else:
            st.metric("Bull level", "no swing high")

        if diag["bear_level"] is not None:
            bear_ok = diag["bear_bars_below"] >= min_bars
            st.metric(
                f"Bear level ({diag['bear_bars_below']}/{min_bars} bars below)",
                f"{diag['bear_level']:.4f}",
                delta="TRIGGERED" if bear_ok else "not triggered",
                delta_color="inverse" if bear_ok else "off",
            )
            if diag["bear_strength"] > -np.inf:
                st.metric("Bear strength", f"{diag['bear_strength']:.5f}")
        else:
            st.metric("Bear level", "no swing low")

    # ── Keyboard hint ─────────────────────────────────────────────────────────
    st.caption("Tip: use the number input to jump directly to any sample index.")


if __name__ == "__main__":
    main()
