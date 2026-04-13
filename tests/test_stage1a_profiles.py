from __future__ import annotations

import pytest

from msb_repr.stage1a.profiles import get_stage1a_profile


def test_get_stage1a_profile_core4():
    assert get_stage1a_profile("core4") == [
        "BTC_USDT_15m",
        "ETH_USDT_15m",
        "SOL_USDT_15m",
        "XRP_USDT_15m",
    ]


def test_get_stage1a_profile_unknown_raises():
    with pytest.raises(ValueError):
        get_stage1a_profile("unknown-profile")
