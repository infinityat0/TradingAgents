"""Tests for the rate-limit retry wrapper."""
import pytest
from unittest.mock import MagicMock

from tradingagents.graph.rate_limit import (
    is_rate_limit_error,
    parse_retry_delay,
    make_retry_wrapper,
)


# ---------------------------------------------------------------------------
# is_rate_limit_error
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("msg", [
    "429 RESOURCE_EXHAUSTED",
    "You exceeded your quota (429)",
    "RATE_LIMIT exceeded",
    "Too Many Requests",
    "Quota exceeded",
])
def test_is_rate_limit_true(msg):
    assert is_rate_limit_error(Exception(msg))


@pytest.mark.unit
@pytest.mark.parametrize("msg", [
    "500 Internal Server Error",
    "Connection refused",
    "Invalid API key",
])
def test_is_rate_limit_false(msg):
    assert not is_rate_limit_error(Exception(msg))


# ---------------------------------------------------------------------------
# parse_retry_delay
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_parse_retry_delay_from_json_style():
    exc = Exception("retryDelay: '11s' in details")
    assert parse_retry_delay(exc) == pytest.approx(13.0)  # 11 + 2 buffer


@pytest.mark.unit
def test_parse_retry_delay_from_prose():
    exc = Exception("Please retry in 20.5s after quota reset")
    assert parse_retry_delay(exc) == pytest.approx(22.5)


@pytest.mark.unit
def test_parse_retry_delay_default():
    exc = Exception("RESOURCE_EXHAUSTED with no delay hint")
    assert parse_retry_delay(exc, default=30.0) == 30.0


# ---------------------------------------------------------------------------
# make_retry_wrapper
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_no_retry_on_success():
    node_fn = MagicMock(return_value={"ok": True})
    wrapped = make_retry_wrapper(node_fn, max_attempts=3)
    result = wrapped({})
    assert result == {"ok": True}
    assert node_fn.call_count == 1


@pytest.mark.unit
def test_no_retry_on_non_rate_limit_error():
    node_fn = MagicMock(side_effect=ValueError("Internal error"))
    wrapped = make_retry_wrapper(node_fn, max_attempts=3)
    with pytest.raises(ValueError):
        wrapped({})
    assert node_fn.call_count == 1


@pytest.mark.unit
def test_retries_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr("tradingagents.graph.rate_limit.time.sleep", MagicMock())
    call_count = 0

    def node_fn(state):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("429 RESOURCE_EXHAUSTED. retryDelay: '1s'")
        return {"done": True}

    wrapped = make_retry_wrapper(node_fn, max_attempts=5)
    result = wrapped({})
    assert result == {"done": True}
    assert call_count == 3


@pytest.mark.unit
def test_raises_after_max_attempts(monkeypatch):
    monkeypatch.setattr("tradingagents.graph.rate_limit.time.sleep", MagicMock())
    node_fn = MagicMock(side_effect=Exception("429 RESOURCE_EXHAUSTED. retryDelay: '1s'"))
    wrapped = make_retry_wrapper(node_fn, max_attempts=3)
    with pytest.raises(Exception, match="RESOURCE_EXHAUSTED"):
        wrapped({})
    assert node_fn.call_count == 3


@pytest.mark.unit
def test_max_attempts_one_is_passthrough():
    node_fn = MagicMock(return_value={})
    assert make_retry_wrapper(node_fn, max_attempts=1) is node_fn
