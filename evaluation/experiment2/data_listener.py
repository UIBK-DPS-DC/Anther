import serial

# This small script was generated using Gemini on the 30.08.2026 in a quick google search. It was then slightly modified.

ser = serial.Serial('/dev/ttyACM0', 921600, timeout=1)
# ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

with open('output.csv', 'w', buffering=1) as f:
    f.write("timestamp (us),current (mA)\n")

    while True:
        raw = ser.readline()

        if not raw:
            continue

        line = raw.decode('utf-8', errors='replace').strip()

        f.write(line + '\n')
        print(line)