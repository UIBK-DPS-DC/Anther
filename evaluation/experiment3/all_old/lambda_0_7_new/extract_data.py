import re
import csv
import sys


def timestamp_to_seconds(hour, minute, second):
    """Convert HH:MM:SS to seconds since midnight."""
    return hour * 3600 + minute * 60 + second


def extract_transmissions(log_text):
    """Extract one record from each [ADR-OVERRIDE] block."""

    # [ADR-OVERRIDE] marks the beginning of every transmission
    blocks = log_text.split("[ADR-OVERRIDE]")

    records = []

    for block in blocks[1:]:
        # ---------------------------------------------------------
        # 1. Planned datarate: experiment_adr_override=5
        # ---------------------------------------------------------
        m = re.search(r"experiment_adr_override\s*=\s*(\d+)", block)
        if not m:
            continue

        datarate = int(m.group(1))

        # ---------------------------------------------------------
        # 2. Used datarate: UL DR / SF : 5 / 7
        # ---------------------------------------------------------
        m = re.search(r"UL DR / SF\s*:\s*(\d+)\s*/", block)
        if not m:
            continue

        used_datarate = int(m.group(1))

        # ---------------------------------------------------------
        # 3. Utility A
        # ---------------------------------------------------------
        m = re.search(r"Utility A\s*:\s*([0-9.eE+-]+)", block)
        if not m:
            continue

        utility_A = float(m.group(1))

        # ---------------------------------------------------------
        # 4. Utility C
        # ---------------------------------------------------------
        m = re.search(r"Utility C\s*:\s*([0-9.eE+-]+)", block)
        if not m:
            continue

        utility_C = float(m.group(1))

        # ---------------------------------------------------------
        # 5. Best class
        # ---------------------------------------------------------
        m = re.search(r"Best class\s*:\s*([AC])", block)
        if not m:
            continue

        best_class = m.group(1)

        # ---------------------------------------------------------
        # 6. Timestamp
        # ---------------------------------------------------------
        m = re.search(
            r"Hour:\s*(\d+),\s*minute:\s*(\d+),\s*second:\s*(\d+)",
            block
        )
        if not m:
            continue

        hour = int(m.group(1))
        minute = int(m.group(2))
        second = int(m.group(3))

        timestamp = timestamp_to_seconds(hour, minute, second)

        records.append({
            "timestamp": timestamp,
            "datarate": datarate,
            "used datarate": used_datarate,
            "utility_A": utility_A,
            "utility_C": utility_C,
            "best_class": best_class,
        })

    return records


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_log.py input.log output.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r", encoding="utf-8") as f:
        log_text = f.read()

    records = extract_transmissions(log_text)

    fieldnames = [
        "timestamp",
        "datarate",
        "used datarate",
        "utility_A",
        "utility_C",
        "best_class",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Extracted {len(records)} transmissions.")
    print(f"Written to: {output_file}")


if __name__ == "__main__":
    main()