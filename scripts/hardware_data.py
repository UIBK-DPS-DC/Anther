"""Read the experiment logs and INA226 recordings without substituting models."""

import csv
import json
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def timestamp_step(previous, current, *, unwrapped=False):
    """Return (elapsed raw units, wrap count), or None for a corrupt jump."""
    delta = current - previous
    if not unwrapped and delta < 0 and 0 < delta + 2**32 <= 1_000_000:
        return delta + 2**32, 1
    if 0 < delta <= (1.0 if unwrapped else 1_000_000):
        return delta, 0
    return None


@dataclass
class PowerTrace:
    time_s: np.ndarray
    energy_j: np.ndarray
    start_s: float
    end_s: float
    samples: int
    rejected_rows: int
    timestamp_wraps: int


def resolve_recording(path):
    """Use a fetched LFS object directly, leaving the working-tree pointer alone."""
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(256)
    if not header.startswith(LFS_HEADER):
        return path
    match = re.search(rb"oid sha256:([0-9a-f]{64})", header)
    if match is None:
        raise ValueError(f"Invalid LFS pointer: {path}")
    oid = match.group(1).decode()
    cached = ROOT_DIR / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    if not cached.is_file():
        relative = path.resolve().relative_to(ROOT_DIR) if path.resolve().is_relative_to(ROOT_DIR) else path
        raise FileNotFoundError(
            f"Missing hardware recording: {path}. Fetch it with "
            f"git lfs fetch origin --include='{relative}' --exclude=''"
        )
    return cached


def read_transfer_log(path):
    text = Path(path).read_text(errors="replace")
    starts = re.findall(r"\[FILE\]\s+(\d+)B\s+CRC32=.*?lambda=([0-9.]+)", text)
    finishes = re.findall(r"\[FILE\] Done:.*?,\s*([0-9.]+)s,\s*class=([AC])", text)
    if len(starts) != 1 or len(finishes) != 1:
        raise ValueError(f"Expected exactly one completed transfer in {path}")
    return {
        "payload_bytes": int(starts[0][0]),
        "lambda": float(starts[0][1]),
        "transfer_duration_s": float(finishes[0][0]),
        "final_class": finishes[0][1],
        "class_sequence": re.findall(r"\[CTRL\] Switched to Class ([AC])", text),
    }


def read_windows(path=None):
    """Optional {recording-relative-path: {start_s, end_s}} in sensor seconds."""
    if path is None:
        return {}
    return json.loads(Path(path).read_text())


def read_power_trace(path, *, voltage=5.0, window=None, bin_seconds=1.0):
    """Integrate V * |I| with trapezoids before reducing points for plotting.

    Raw serial files begin with stale bytes from the previous recording. Require
    32 consecutive plausible samples within two seconds of the Arduino reset
    before starting. Reject malformed rows, isolated timestamp jumps, and current
    magnitudes above 250 mA. Unwrap micros() only for a plausible 32-bit wrap.
    Already-cropped files have a trustworthy, unwrapped ``time (s)`` column.

    No transfer boundary is inferred from the power waveform. Without an explicit
    window this returns the complete cleaned recording, INCLUDING setup/reset.
    """
    if voltage <= 0 or bin_seconds <= 0:
        raise ValueError("Voltage and bin width must be positive")
    start = float(window["start_s"]) if window else -math.inf
    end = float(window["end_s"]) if window else math.inf
    if not start < end:
        raise ValueError(f"Invalid measurement window: {window}")

    points_t, points_e = [], []
    rejected = wraps = samples = 0
    pending = deque(maxlen=32)
    last_raw = None
    offset = 0.0
    previous = None
    total_energy = 0.0
    first_time = last_time = None
    next_bin = None

    def integrate(timestamp, current):
        nonlocal previous, total_energy, first_time, last_time, samples, next_bin
        power = voltage * abs(current) / 1000.0
        if start <= timestamp <= end:
            samples += 1
        if previous is not None:
            old_t, old_power = previous
            left, right = max(old_t, start), min(timestamp, end)
            if right > left:
                slope = (power - old_power) / (timestamp - old_t)
                left_power = old_power + slope * (left - old_t)
                right_power = old_power + slope * (right - old_t)
                if first_time is None:
                    first_time = left
                    points_t.append(0.0)
                    points_e.append(0.0)
                    next_bin = bin_seconds
                total_energy += 0.5 * (left_power + right_power) * (right - left)
                last_time = right
                if right - first_time >= next_bin:
                    points_t.append(right - first_time)
                    points_e.append(total_energy)
                    next_bin = right - first_time + bin_seconds
        previous = (timestamp, power)

    with resolve_recording(path).open(errors="replace", newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        unwrapped = "time (s)" in columns
        time_index = columns.index("time (s)") if unwrapped else columns.index("timestamp (us)")
        current_index = columns.index("current (mA)")
        started = unwrapped
        for row in reader:
            try:
                raw = float(row[time_index])
                current = float(row[current_index])
                if (not unwrapped and len(row) != 2) or not (
                    math.isfinite(raw) and math.isfinite(current)
                    and raw >= 0 and abs(current) <= 250
                ):
                    raise ValueError
                if not unwrapped and (raw >= 2**32 or not raw.is_integer()):
                    raise ValueError
            except (ValueError, IndexError):
                rejected += 1
                continue
            if not started:
                if raw > 2_000_000:
                    rejected += 1
                    continue
                if pending and not 0 < raw - pending[-1][0] < 100_000:
                    rejected += len(pending)
                    pending.clear()
                pending.append((raw, current))
                if len(pending) < 32:
                    continue
                started = True
                for timestamp, value in pending:
                    integrate(timestamp / 1e6, value)
                last_raw = raw
                continue
            if last_raw is not None:
                step = timestamp_step(last_raw, raw, unwrapped=unwrapped)
                if step is None:
                    rejected += 1
                    continue
                offset += step[1] * 2**32
                wraps += step[1]
            timestamp = raw if unwrapped else (raw + offset) / 1e6
            integrate(timestamp, current)
            last_raw = raw
            if timestamp >= end:
                break

    if first_time is None or last_time is None or last_time <= first_time:
        raise ValueError(f"No usable measurement interval in {path}")
    if window and (first_time > start + 0.01 or last_time < end - 0.01):
        raise ValueError(f"Recording does not cover requested window {window}: {path}")
    if points_t[-1] != last_time - first_time:
        points_t.append(last_time - first_time)
        points_e.append(total_energy)
    return PowerTrace(
        np.array(points_t), np.array(points_e), first_time, last_time,
        samples, rejected, wraps,
    )
