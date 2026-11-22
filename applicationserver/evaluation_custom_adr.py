"""Optional, evaluation-only ADR override schedule.

This module is deliberately independent from the FUOTA sender. Production
transfers should leave the evaluation hook disabled.
"""

import time
from collections import deque
from collections.abc import Callable
from statistics import fmean

import requests


ADR_OVERRIDE_VARIABLE = "experiment_adr_override"


def get_device_via_rest(
    dev_eui: str, rest_server: str, token: str, timeout: float = 10.0
) -> dict:
    url = f"{rest_server.rstrip('/')}/api/devices/{dev_eui}"
    response = requests.get(
        url, headers={"Authorization": f"Bearer {token}"}, timeout=timeout
    )
    response.raise_for_status()
    return response.json()


def set_device_variable_via_rest(
    dev_eui: str,
    key: str,
    value: str,
    rest_server: str,
    token: str,
    timeout: float = 10.0,
) -> dict:
    current = get_device_via_rest(dev_eui, rest_server, token, timeout)
    device = current.get("device")
    if device is None:
        raise RuntimeError(
            f"Unexpected response shape from GET {rest_server}/api/devices/{dev_eui}: "
            "no 'device' key"
        )

    variables = dict(device.get("variables") or {})
    variables[key] = str(value)
    device["variables"] = variables

    url = f"{rest_server.rstrip('/')}/api/devices/{dev_eui}"
    response = requests.put(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={"device": device},
        timeout=timeout,
    )
    response.raise_for_status()
    return device


def trigger_adr_override(
    dev_eui: str,
    value: int,
    rest_server: str,
    token: str,
) -> bool:
    try:
        set_device_variable_via_rest(
            dev_eui, ADR_OVERRIDE_VARIABLE, str(value), rest_server, token
        )
    except requests.RequestException as error:
        print(f"[ADR-OVERRIDE] Failed to set device variable: {error}")
        return False
    print(f"[ADR-OVERRIDE] Set {ADR_OVERRIDE_VARIABLE}={value} on {dev_eui}")
    return True


class AdrOverrideSchedule:
    """Repeat: DR5 for 5 minutes, descend to DR0 by 30 minutes, ascend by 60."""

    def __init__(
        self,
        apply: Callable[[int], bool],
        start: float | None = None,
        period: float = 3600.0,
    ):
        self._apply = apply
        self._start = time.monotonic() if start is None else start
        self._period = period
        self._last_dr: int | None = None

    def target_dr(self, now: float | None = None) -> int:
        elapsed = ((time.monotonic() if now is None else now) - self._start) % self._period
        # The following are manually derived linear segment functions
        if elapsed < 300.0:
            return 5
        if elapsed <= 1800.0:
            return int(-((3)/(1500)) * elapsed + 5.6)
        if elapsed <= 2100.0:
            return 2
        return int(((3)/(1500)) * elapsed - 2.2)

    def update(self, now: float | None = None) -> int:
        dr = self.target_dr(now)
        if dr != self._last_dr:
            if self._apply(dr):
                self._last_dr = dr
        return dr


class HistoricalMovingAverageADR:
    """Predict the next DR from the most recent observed DR history.

    The observation source represents the actual network ADR state. The
    predictor never includes the current observation in the prediction, so
    each result is based only on data that would have been available before
    the next transmission.
    """

    def __init__(
        self,
        apply: Callable[[int], bool],
        observation: Callable[[float | None], int],
        window: int = 5,
        initial_dr: int = 5,
    ):
        if window < 1:
            raise ValueError("window must be at least 1")
        if not 0 <= initial_dr <= 5:
            raise ValueError("initial_dr must be between DR0 and DR5")

        self._apply = apply
        self._observation = observation
        self._history = deque[int](maxlen=window)
        self._initial_dr = initial_dr
        self._last_prediction: int | None = None

    @staticmethod
    def _round_dr(value: float) -> int:
        # DR is ordinal: average the observed DR indices, then quantize to
        # the nearest valid EU868 DR using half-up rounding.
        return max(0, min(5, int(value + 0.5)))

    def update(self, now: float | None = None) -> int:
        if self._history:
            prediction = self._round_dr(fmean(self._history))
        else:
            prediction = self._initial_dr

        observed_dr = self._observation(now)
        if not 0 <= observed_dr <= 5:
            raise ValueError(f"Unsupported observed DR: {observed_dr}")
        self._history.append(observed_dr)

        if prediction != self._last_prediction:
            self._apply(prediction)
            self._last_prediction = prediction

        return prediction
