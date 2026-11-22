#!/usr/bin/env bash

# This script is used to execute multiple downlinks in squence.
# The structure is to run downlinks with Class A and Class C in combination with different amounts of data.
# The aim of this is to compare the measured energy and time consumption with the predicted once.
# Additionally the script uses class switching to easily mark timestamps in the power consumption data. Needs to be manually cut.
# 1 2 5 10 50 100
# Generate nice file sizes, with random content. 
cd ~/deltafuota/
dd if=/dev/urandom of=evaluation/experiment2/dr2/spacer.bin bs=1 count=10
python3 networkserver/application_server.py switch-class --class A
python3 networkserver/application_server.py switch-class --class C
python3 networkserver/application_server.py set-interval --seconds 24
for size in 1 2 5 10 50 100; do
    dd if=/dev/urandom \
       of="evaluation/experiment2/dr2/random_${size}KiB.bin" \
       bs=1K count="$size" \
       status=none
    python3 networkserver/application_server.py switch-class --class A
    python3 networkserver/application_server.py send-file --file evaluation/experiment2/dr2/random_${size}KiB.bin >> evaluation/experiment2/dr2/prediction_${size}KiB.log
    python3 networkserver/application_server.py switch-class --class C
    python3 networkserver/application_server.py send-file --file evaluation/experiment2/dr2/spacer.bin # This is to fill the initial RX1 window of the Uplink
    python3 networkserver/application_server.py send-file --file evaluation/experiment2/dr2/random_${size}KiB.bin
done
