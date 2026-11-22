#!/bin/bash

set -e

# Raspberry Pi
PI_USER="dps"
PI_HOST="192.168.178.38"
PI_APP_DIR="/home/dps/deltafuota/applicationserver"

# PC paths
EXP_DIR="."
LOG_DIR="${EXP_DIR}/logs"
MEASUREMENTS_DIR="${EXP_DIR}/measurements"
MEASUREMENT_SCRIPT="${MEASUREMENTS_DIR}/measure.py"

for size in 1 2 5 10 50; do
    dd if=/dev/urandom \
    of="../experiment3/random_${size}KiB.bin" \
    bs=1K count="$size" \
    status=none
    scp ../experiment3/random_${size}KiB.bin "${PI_USER}@${PI_HOST}:${PI_APP_DIR}/../evaluation/experiment3"
done

for lambda in 0.7 0.1 0.4 0.65; do
    mkdir -p "${LOG_DIR}/lambda_${lambda}"
    mkdir -p "${MEASUREMENTS_DIR}/lambda_${lambda}"
    for size in 1 2 5 10 50; do

        # Output files on the PC
        SEND_LOG="${LOG_DIR}/lambda_${lambda}/dynamic_${size}KiB.log"
        MEASUREMENT_OUTPUT="${MEASUREMENTS_DIR}/lambda_${lambda}/dynamic_${size}KiB_measurement.csv"

        echo "========================================"
        echo "Starting experiment"
        echo "Lambda: ${lambda}"
        echo "Size:   ${size} KiB"
        echo "========================================"        

        # 0. Wait for empty queue
        ssh "${PI_USER}@${PI_HOST}" "
            cd '${PI_APP_DIR}' &&
            ../.venv/bin/python3 application_server.py mqtt-wait-empty-queue
        " >> "${SEND_LOG}" 2>&1

        # 1. Start measurement on the PC
        echo "Starting measurement..."

        python3 "${MEASUREMENT_SCRIPT}" "${MEASUREMENT_OUTPUT}" &

        MEASUREMENT_PID=$!
        sleep 3 # Give the measurement script a moment to open the serial port

        # 2. Marker in power measurement
        ssh "${PI_USER}@${PI_HOST}" "
            cd '${PI_APP_DIR}' &&
            ../.venv/bin/python3 application_server.py switch-class --class A &&
            ../.venv/bin/python3 application_server.py switch-class --class C &&
            ../.venv/bin/python3 application_server.py switch-class --class A
        "

        # 3. Execute send-file on the Raspberry Pi
        
        echo "Starting file transfer..."
        date
        SECONDS=0

        ssh "${PI_USER}@${PI_HOST}" "
            cd '${PI_APP_DIR}' &&
            ../.venv/bin/python3 application_server.py send-file \
                --file '../evaluation/experiment3/random_${size}KiB.bin' \
                --l '${lambda}' \
                --adr-override \
                --mqtt-timeout 3000
        " >> "${SEND_LOG}" 2>&1

        echo "File transfer finished. Elapsed: $SECONDS seconds"
        date

        # 4. Reset the Pi for the next experiment
        echo "Resetting datarate and classes..."

        ssh "${PI_USER}@${PI_HOST}" "
            cd '${PI_APP_DIR}' &&
            ../.venv/bin/python3 application_server.py set-datarate --datarate 5 &&
            ../.venv/bin/python3 application_server.py switch-class --class A &&
            ../.venv/bin/python3 application_server.py switch-class --class C &&
            ../.venv/bin/python3 application_server.py switch-class --class A
        "

        # 5. Wait for empty queue
        ssh "${PI_USER}@${PI_HOST}" "
            cd '${PI_APP_DIR}' &&
            ../.venv/bin/python3 application_server.py mqtt-wait-empty-queue
        " >> "${SEND_LOG}" 2>&1

        # 6. Stop measurement cleanly
        echo "Stopping measurement..."

        kill -9 "${MEASUREMENT_PID}"

        # Wait for the measurement script to finish and close the CSV
        wait "${MEASUREMENT_PID}" || true

        echo "Measurement saved to:"
        echo "  ${MEASUREMENT_OUTPUT}"
        echo "Experiment finished."
        echo

    done
done

echo "========================================"
echo "All experiments finished."
echo "========================================"
