#!/usr/bin/env bash

# This script is used to execute multiple downlinks in squence.
# The structure is to run downlinks with Class A and Class C in combination with different amounts of data.
# The aim of this is to compare the measured energy and time consumption with the predicted once.
# Additionally the script uses class switching to easily mark timestamps in the power consumption data. Needs to be manually cut.
# 1 2 5 10 50 100
# Generate nice file sizes, with random content. 
cd ~/deltafuota/networkserver
python3 -c  'from application_server_claude import save_parameters, EnergyParameters; save_parameters(EnergyParameters(n=5, uplink_payload_bytes=5, downlink_payload_bytes=218, report_interval=5.0, class_c_fragment_interval=3.5, ul_dr=5, dl_dr=5, duty_cycle=0.01, dl_duty_cycle=0.10, lambda_=0.5))'
mv ~/deltafuota/networkserver/configuration_cache_e21ca312cb69528a.json ~/deltafuota/
cd ~/deltafuota/
python3 networkserver/application_server.py switch-class --class A
python3 networkserver/application_server.py switch-class --class C
for size in 1 2 5; do
    dd if=/dev/urandom \
       of="evaluation/experiment2/dr5/random_${size}KiB.bin" \
       bs=1K count="$size" \
       status=none
    python3 networkserver/application_server.py switch-class --class A
    python3 networkserver/application_server.py send-file --file evaluation/experiment2/dr5/random_${size}KiB.bin >> evaluation/experiment2/dr5/prediction_${size}KiB.log
    python3 networkserver/application_server.py switch-class --class C
    python3 networkserver/application_server.py send-file --file evaluation/experiment2/dr5/spacer.bin # This is to fill the initial RX1 window of the Uplink
    python3 networkserver/application_server.py send-file --file evaluation/experiment2/dr5/random_${size}KiB.bin
done
