import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

# CSV_FILE = "City_center_Long-Term_self.csv"
CSV_FILE = "BUT_Long-Term.csv"

# Time windows to search
WINDOWS_HOURS = [1, 3, 6, 12]

# Desired scenario
TARGET_START_SNR = -7
TARGET_END_SNR = -13

# How close the SNR values need to be to the target
SNR_TOLERANCE = 2.0

# RSSI range that we consider "good enough"
# Adjust this to your gateway/device setup.
MIN_RSSI = -125

# Plot data around the detected event
PLOT_BEFORE_HOURS = 6
PLOT_AFTER_HOURS = 6


# ============================================================
# LoRaWAN data-rate information
# EU868 example
# ============================================================

# DR : SF
DR_TO_SF = {
    0: 12,
    1: 11,
    2: 10,
    3: 9,
    4: 8,
    5: 7,
}

# Approximate SNR limits commonly used for ADR calculations.
# These are nominal required SNR values and should be adapted
# if you use a particular LoRaWAN implementation.
SF_REQUIRED_SNR = {
    7: -7.5,
    8: -10.0,
    9: -12.5,
    10: -15.0,
    11: -17.5,
    12: -20.0,
}


def snr_to_best_dr(snr):
    """
    Return the highest data rate (largest DR number)
    that should theoretically work for the given SNR.

    This is a simplified calculation and does NOT include
    ADR margin or implementation-specific rules.
    """

    possible = []

    for dr, sf in DR_TO_SF.items():
        required_snr = SF_REQUIRED_SNR[sf]

        if snr >= required_snr:
            possible.append(dr)

    if not possible:
        return None

    return max(possible)


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(CSV_FILE)

df["datetime"] = pd.to_datetime(
    df["time"],
    unit="s",
    utc=True
)

# Sort chronologically
df = df.sort_values("datetime").reset_index(drop=True)

print(f"Loaded {len(df)} measurements")
print(f"From: {df['datetime'].min()}")
print(f"To:   {df['datetime'].max()}")


# ============================================================
# Find largest SNR drops
# ============================================================

def find_largest_drops(df, window_hours):

    window = pd.Timedelta(hours=window_hours)

    events = []

    times = df["datetime"].values

    for i in range(len(df)):

        current_time = df.loc[i, "datetime"]
        current_snr = df.loc[i, "snr"]

        # Measurements that occurred before this measurement
        mask = (
            (df["datetime"] < current_time) &
            (df["datetime"] >= current_time - window)
        )

        previous = df.loc[mask]

        if len(previous) == 0:
            continue

        # Find the highest SNR before the current measurement.
        # We are interested in a drop, so use the maximum SNR.
        j = previous["snr"].idxmax()

        start_snr = df.loc[j, "snr"]
        start_time = df.loc[j, "datetime"]

        drop = start_snr - current_snr

        events.append({
            "start_time": start_time,
            "end_time": current_time,
            "duration_hours":
                (current_time - start_time).total_seconds() / 3600,
            "start_snr": start_snr,
            "end_snr": current_snr,
            "snr_drop": drop,
            "start_rssi": df.loc[j, "rssi"],
            "end_rssi": df.loc[i, "rssi"],
        })

    result = pd.DataFrame(events)

    if len(result) == 0:
        return result

    # Largest drops first
    result = result.sort_values(
        "snr_drop",
        ascending=False
    )

    return result


# ============================================================
# Search all windows
# ============================================================

all_events = []

for hours in WINDOWS_HOURS:

    events = find_largest_drops(df, hours)

    if len(events):

        events["window_hours"] = hours

        all_events.append(events)


if not all_events:
    print("No events found.")
    exit()


events = pd.concat(all_events, ignore_index=True)


# Remove duplicate-ish events.
# Keep the largest drop for each end timestamp.
events = events.sort_values(
    "snr_drop",
    ascending=False
)

events = events.drop_duplicates(
    subset=["end_time"],
    keep="first"
)


# ============================================================
# Display largest drops
# ============================================================

print("\n" + "=" * 100)
print("LARGEST SNR DROPS")
print("=" * 100)

columns = [
    "window_hours",
    "duration_hours",
    "start_snr",
    "end_snr",
    "snr_drop",
    "start_rssi",
    "end_rssi",
    "start_time",
    "end_time",
]

for hour_window in WINDOWS_HOURS:

    print("\n" + "-" * 100)
    print(f"Largest drops for {hour_window} hour window")
    print("-" * 100)

    window_events = events[
        events["window_hours"] == hour_window
    ]

    if len(window_events) == 0:
        print("No events found.")
        continue

    # if hour_window == WINDOWS_HOURS[0]:
    #     print(
    #         window_events[columns].to_string(index=False)
    #     )
    # else:
    print(
        window_events[columns].head(20).to_string(index=False)
    )
    if len(window_events) > 20:
        print(f"... and {len(window_events) - 20} more events")
    print(f"Total events for {hour_window} hour window found: {len(window_events)}")


# ============================================================
# Find events resembling -7 dB -> -13 dB
# ============================================================

target_events = events[
    (events["start_snr"] > -7)
    &
    (events["end_snr"] > -15) & (events["end_snr"] < -12.5) 
    &
    (events["end_rssi"] >= MIN_RSSI)
].copy()


print("\n" + "=" * 100)
print("POTENTIAL DR5 -> DR2 SCENARIOS")
print("=" * 100)

if len(target_events) == 0:

    print("No events matching the requested scenario were found.")

else:

    # Calculate theoretical best DR
    target_events["start_dr"] = target_events["start_snr"].apply(
        snr_to_best_dr
    )

    target_events["end_dr"] = target_events["end_snr"].apply(
        snr_to_best_dr
    )

    print(
        target_events[
            [
                "window_hours",
                "duration_hours",
                "start_snr",
                "end_snr",
                "snr_drop",
                "start_rssi",
                "end_rssi",
                "start_dr",
                "end_dr",
                "start_time",
                "end_time",
            ]
        ].head(20).to_string(index=False)
    )


# ============================================================
# Select best candidate for plotting
# ============================================================

if len(target_events) > 0:
    best = target_events.iloc[0]

else:
    # Fall back to the largest SNR drop
    best = events.iloc[0]

best = events[
    (abs(events["start_time"] == pd.Timestamp("2020-03-04 11:23:28.513499975+00:00"))) 
    &
    (abs(events["end_time"] == pd.Timestamp("2020-03-04 11:49:04.531470059+00:00")))
].iloc[0]

start_time = best["start_time"]
end_time = best["end_time"]

plot_start = start_time - pd.Timedelta(
    hours=PLOT_BEFORE_HOURS
)

plot_end = end_time + pd.Timedelta(
    hours=PLOT_AFTER_HOURS
)

plot_df = df[
    (df["datetime"] >= plot_start)
    &
    (df["datetime"] <= plot_end)
].copy()


# ============================================================
# Print selected event
# ============================================================

print("\n" + "=" * 100)
print("SELECTED EVENT")
print("=" * 100)

print(f"Start:       {start_time}")
print(f"End:         {end_time}")
print(f"Duration:    {best['duration_hours']:.2f} h")
print(f"SNR:         {best['start_snr']:.1f} -> {best['end_snr']:.1f} dB")
print(f"SNR change:  -{best['snr_drop']:.1f} dB")
print(f"RSSI:        {best['start_rssi']:.1f} -> {best['end_rssi']:.1f} dBm")

print(
    f"Approx. DR:  "
    f"{snr_to_best_dr(best['start_snr'])} -> "
    f"{snr_to_best_dr(best['end_snr'])}"
)

DR_FOR_MIN_SNR = [
    (-7.5, 5),
    (-10.0, 4),
    (-12.5, 3),
    (-15.0, 2),
]


start_time = plot_df["datetime"].tolist()[2]
for (date, snr) in zip(plot_df["datetime"], plot_df["snr"]):
    if date < start_time:
        continue
    date_delta = date - start_time
    d = pd.Timestamp(0) + date_delta
    for min_snr, dr in DR_FOR_MIN_SNR:
        if snr > min_snr:
            print(f"{d.hour}:{d.minute}:{d.second}: {snr} dB (DR{dr})")
            break

# ============================================================
# Plot
# ============================================================

fig, ax1 = plt.subplots(figsize=(14, 7))

# SNR
ax1.plot(
    plot_df["datetime"],
    plot_df["snr"],
    marker=".",
    linewidth=1,
    label="SNR"
)

# ax1.axhline(
#     -7.5,
#     linestyle="--",
#     label="SF7 / DR5 required SNR"
# )

# ax1.axhline(
#     -15,
#     linestyle="--",
#     label="SF10 / DR2 required SNR"
# )

ax1.scatter(
    [start_time, end_time],
    [best["start_snr"], best["end_snr"]],
    s=100,
    zorder=5
)

ax1.set_ylabel("SNR [dB]")
ax1.set_xlabel("Time")
ax1.grid(True, alpha=0.3)

# RSSI on second axis
ax2 = ax1.twinx()

ax2.plot(
    plot_df["datetime"],
    plot_df["rssi"],
    marker=".",
    linewidth=1,
    alpha=0.6,
    label="RSSI",
    color="orange"
)

ax2.set_ylabel("RSSI [dBm]")

# Event line
ax1.axvline(
    start_time,
    linestyle=":",
    linewidth=2
)

ax1.axvline(
    end_time,
    linestyle=":",
    linewidth=2
)

ax1.set_title(
    "Large LoRaWAN SNR Drop\n"
    f"{best['start_snr']:.1f} dB → {best['end_snr']:.1f} dB "
    f"over {best['duration_hours']:.2f} h"
)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="best"
)

plt.tight_layout()
plt.show()