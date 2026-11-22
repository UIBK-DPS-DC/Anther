import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, sys

# Generated using ChatGPT on the 30.08.2026 using the base code written by hand as a template, with this prompt:
# Modify this script, so it is able to split the data from the original csv into as many different csvs as I need based on intervals, that I can type in while looking at the plot.


# ============================================================
# Configuration
# ============================================================

# Number of rows to process from the original CSV.
# Set to None to process the entire file.
MAX_ROWS = None

# Supply voltage
VOLTAGE = 5.0


def time_power(input_file, output_dir):
    # ============================================================
    # Load data
    # ============================================================

    df = pd.read_csv(input_file)

    # Keep original data intact, but create numeric columns for processing
    df["timestamp"] = pd.to_numeric(df["timestamp (us)"], errors="coerce")
    df["current"] = pd.to_numeric(df["current (mA)"], errors="coerce")

    df = df.dropna(subset=["timestamp", "current"]).copy()

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS).copy()


    # ============================================================
    # Unwrap 32-bit timestamp
    # ============================================================

    # 32-bit unsigned timestamp wraps at 2^32.
    # timestamp is expressed in units of 100 us.

    UINT32_MAX_PLUS_ONE = 2**32

    # Detect timestamp overflow
    wraps = (df["timestamp"].diff() < 0).cumsum()

    # Unwrap timestamp
    df["timestamp"] = (
        df["timestamp"] + wraps * UINT32_MAX_PLUS_ONE
    )

    # Convert 100 us units to seconds
    df["time (s)"] = df["timestamp"] / 1e6


    # ============================================================
    # Calculate power
    # ============================================================

    # Power in mW = V * mA
    df["power (mW)"] = VOLTAGE * df["current"].abs()

    # Remove zero-power samples
    df = df[df["power (mW)"] > 0].copy()


    # ============================================================
    # Plot
    # ============================================================

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        df["time (s)"],
        df["power (mW)"],
        linewidth=1.0
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (mW)")
    ax.set_title("Power Consumption vs. Time")

    ax.set_yscale("log")

    ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10, subs=range(2, 10))
    )


    def log_tick_formatter(value, pos):
        return f"{value:g}"


    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(log_tick_formatter)
    )

    ax.yaxis.set_minor_formatter(
        ticker.FuncFormatter(log_tick_formatter)
    )

    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    # Keep plot open while asking for input
    plt.show(block=False)
    plt.pause(0.1)


    # ============================================================
    # Create output directory
    # ============================================================

    os.makedirs(output_dir, exist_ok=True)


    # ============================================================
    # Ask how many intervals
    # ============================================================

    print()
    print("=" * 60)
    print("DATA SPLITTING")
    print("=" * 60)

    print(f"Available time range:")
    print(f"  Start: {df['time (s)'].min():.6f} s")
    print(f"  End:   {df['time (s)'].max():.6f} s")
    print()

    num_intervals = int(
        input("How many intervals do you want to create? ")
    )


    # ============================================================
    # Process intervals
    # ============================================================

    results = []

    for i in range(1, num_intervals + 1):

        print()
        print("-" * 60)
        print(f"INTERVAL {i}")
        print("-" * 60)

        print("Look at the plot and enter the START time in seconds.")
        x = input("Start time: ").strip()

        if x == "":
            print("Empty start time entered. Using 0.")
            start_time = 0.0
        else:
            start_time = float(x)

        print("Enter the END time in seconds.")
        x = input("End time: ").strip()

        if x == "":
            print("Empty end time entered. Using the end of the data.")
            end_time = df["time (s)"].max()
        else:
            end_time = float(x)

        if end_time <= start_time:
            print("ERROR: End time must be greater than start time.")
            print("Skipping this interval.")
            continue

        # --------------------------------------------------------
        # Select interval
        # --------------------------------------------------------

        interval = df[
            (df["time (s)"] >= start_time) &
            (df["time (s)"] <= end_time)
        ].copy()

        if interval.empty:
            print("WARNING: No data found in this interval.")
            continue

        # --------------------------------------------------------
        # Calculate energy
        # --------------------------------------------------------

        interval["dt (s)"] = interval["time (s)"].diff()

        # The first sample has no previous timestamp.
        # Fill it with zero so it doesn't affect the result.
        interval["dt (s)"] = interval["dt (s)"].fillna(0)

        interval["energy (mJ)"] = (
            interval["power (mW)"] *
            interval["dt (s)"]
        )

        total_energy_mJ = interval["energy (mJ)"].sum()

        duration = end_time - start_time

        # --------------------------------------------------------
        # Ask for filename
        # --------------------------------------------------------

        print()
        print(f"Interval contains {len(interval):,} samples.")
        print(f"Energy: {total_energy_mJ / 1000.0:.6f} J")
        print(f"Time:   {duration:.6f} s")

        filename = input(
            f"Filename for interval {i} "
            f"(empty = interval_{i}.csv): "
        ).strip()

        if filename == "":
            filename = f"interval_{i}.csv"

        # Make sure filename ends in .csv
        if not filename.lower().endswith(".csv"):
            filename += ".csv"

        output_path = os.path.join(
            output_dir,
            filename
        )

        # --------------------------------------------------------
        # Save CSV
        # --------------------------------------------------------

        interval.to_csv(
            output_path,
            index=False
        )

        print(f"Saved: {output_path}")

        # --------------------------------------------------------
        # Store result
        # --------------------------------------------------------

        results.append({
            "interval": i,
            "start_time_s": start_time,
            "end_time_s": end_time,
            "duration_s": duration,
            "samples": len(interval),
            "energy_J": total_energy_mJ / 1000.0,
            "file": output_path
        })


    # ============================================================
    # Print summary
    # ============================================================

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results:

        results_df = pd.DataFrame(results)

        print(
            results_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

        # Also save a summary CSV
        summary_path = os.path.join(
            output_dir,
            "summary.csv"
        )

        results_df.to_csv(
            summary_path,
            index=False
        )

        print()
        print(f"Summary saved to: {summary_path}")

    else:
        print("No intervals were created.")


    # Keep plot available
    print()
    print("Press Enter to close the plot.")
    input()

    plt.close(fig)


def main():
    if len(sys.argv) != 3:
        print("Usage: python time_power.py input.csv output_dir")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    time_power(input_file, output_dir)

if __name__ == "__main__":
    main()