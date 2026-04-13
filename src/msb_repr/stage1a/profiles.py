"""Named dataset profiles for Stage 1A experiments."""

from __future__ import annotations


STAGE1A_DATASET_PROFILES: dict[str, list[str]] = {
    # Mirrors the strong multi-symbol short AE / joint-global-s1 training universe.
    "core4": [
        "BTC_USDT_15m",
        "ETH_USDT_15m",
        "SOL_USDT_15m",
        "XRP_USDT_15m",
    ],
    # Altcoin-heavy set used in broader multi-symbol experiments.
    "altcoin6": [
        "BTC_USDT_15m",
        "ETH_USDT_15m",
        "SOL_USDT_15m",
        "XRP_USDT_15m",
        "NEAR_USDT_15m",
        "INJ_USDT_15m",
    ],
    # Active pipeline symbols only; useful for narrower production-aligned representation tests.
    "pipeline3": [
        "BTC_USDT_15m",
        "XRP_USDT_15m",
        "NEAR_USDT_15m",
    ],
}


def get_stage1a_profile(name: str) -> list[str]:
    try:
        return STAGE1A_DATASET_PROFILES[name].copy()
    except KeyError as exc:
        available = ", ".join(sorted(STAGE1A_DATASET_PROFILES))
        raise ValueError(f"Unknown Stage 1A dataset profile: {name}. Available: {available}") from exc
