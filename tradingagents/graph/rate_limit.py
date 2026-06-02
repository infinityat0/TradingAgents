# TradingAgents/graph/rate_limit.py
"""Rate-limit retry utilities for LangGraph node functions.

When a provider returns HTTP 429 / RESOURCE_EXHAUSTED the response always
includes a suggested retry delay.  The helpers here parse that delay out of
the exception string and retry the node after sleeping for it, so a transient
quota hit doesn't abort an entire multi-hour run.
"""

import logging
import re
import time
from typing import Callable

logger = logging.getLogger(__name__)

# Matches "retryDelay": "11s" (JSON) or retryDelay: '11s' (repr variants)
_RETRY_DELAY_JSON = re.compile(r"retryDelay[\"'\s:]+(\d+(?:\.\d+)?)s")
# Matches "Please retry in 11.867s"
_RETRY_DELAY_MSG = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)

_RATE_LIMIT_MARKERS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "RATE_LIMIT",
    "RATELIMIT",
    "TOO MANY REQUESTS",
    "QUOTA",
)


def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).upper()
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)


def parse_retry_delay(exc: Exception, default: float = 65.0) -> float:
    """Return the provider-suggested retry delay in seconds, or *default*.

    Adds a 2-second safety buffer on top of the parsed value so that the
    quota window has fully rolled over before the retry fires.
    """
    msg = str(exc)
    for pattern in (_RETRY_DELAY_JSON, _RETRY_DELAY_MSG):
        m = pattern.search(msg)
        if m:
            return float(m.group(1)) + 2.0
    return default


def make_retry_wrapper(node_fn: Callable, max_attempts: int) -> Callable:
    """Wrap *node_fn* to retry on rate-limit (429 / RESOURCE_EXHAUSTED) errors.

    Non-rate-limit exceptions propagate immediately.  On the final attempt
    the rate-limit exception also propagates so the caller (LangGraph or the
    user) sees the real error rather than a silent failure.

    Compose with the delay wrapper as ``delay(retry(node_fn))`` so the
    proactive turn delay fires once before all retry attempts, while the
    reactive retry delay fires only between attempts.
    """
    if max_attempts <= 1:
        return node_fn

    def wrapper(state):
        for attempt in range(1, max_attempts + 1):
            try:
                return node_fn(state)
            except Exception as exc:
                if not is_rate_limit_error(exc) or attempt >= max_attempts:
                    raise
                delay = parse_retry_delay(exc)
                logger.warning(
                    "Rate limit hit (attempt %d/%d). "
                    "Sleeping %.0fs before retry…",
                    attempt,
                    max_attempts,
                    delay,
                )
                time.sleep(delay)

    return wrapper
