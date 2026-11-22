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
        # 3. Class A
        # ---------------------------------------------------------

        # Find "Class A" and then extract the first Latency after it
        m = re.search(
            r"Class A.*?Latency\s*:\s*([0-9.eE+-]+)\s*s",
            block,
            re.DOTALL
        )
        if not m:
            continue

        latency_A = float(m.group(1))

        # Find "Class A" and then extract the first Energy after it
        m = re.search(
            r"Class A.*?Energy\s*:\s*([0-9.eE+-]+)\s*J",
            block,
            re.DOTALL
        )
        if not m:
            continue

        energy_A = float(m.group(1))


        # ---------------------------------------------------------
        # 4. Class C
        # ---------------------------------------------------------

        # Find "Class C" and then extract the first Latency after it
        m = re.search(
            r"Class C.*?Latency\s*:\s*([0-9.eE+-]+)\s*s",
            block,
            re.DOTALL
        )
        if not m:
            continue

        latency_C = float(m.group(1))

        # Find "Class C" and then extract the first Energy after it
        m = re.search(
            r"Class C.*?Energy\s*:\s*([0-9.eE+-]+)\s*J",
            block,
            re.DOTALL
        )
        if not m:
            continue

        energy_C = float(m.group(1))

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
            "latency_A": latency_A,
            "energy_A": energy_A,
            "latency_C": latency_C,
            "energy_C": energy_C,
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
        "latency_A",
        "energy_A",
        "latency_C",
        "energy_C",
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