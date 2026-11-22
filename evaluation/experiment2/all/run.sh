#!/usr/bin/env bash

# This script is used to execute multiple downlinks in squence.
# The structure is to run downlinks with Class A and Class C in combination with different amounts of data.
# The aim of this is to compare the measured energy and time consumption with the predicted once.
# Additionally the script uses class switching to easily mark timestamps in the power consumption data. Needs to be manually cut.
# 1 2 5 10 50
# Generate nice file sizes, with random content. 
cd ~/deltafuota/networkserver
python3 application_server_claude.py set-datarate --datarate 5
python3 application_server_claude.py switch-class --class A
python3 application_server_claude.py switch-class --class C
python3 application_server_claude.py switch-class --class A
for size in 1 2 5 10 50 100; do
    dd if=/dev/urandom \
       of="../evaluation/experiment2/all/random_${size}KiB.bin" \
       bs=1K count="$size" \
       status=none
done
echo "Finished setup"
# for size in 1 2 5 10 50; do
#     python3 application_server_claude.py send-file --file ../evaluation/experiment2/all/random_${size}KiB.bin --mqtt-timeout 3000 --class-mode C > ../evaluation/experiment2/all/class_c_${size}KiB.log
#     echo "Finished sending random_${size}KiB.bin with class C"
#     python3 application_server_claude.py switch-class --class A
# done
for size in 1 2 5 10 50; do
    python3 application_server_claude.py send-file --file ../evaluation/experiment2/all/random_${size}KiB.bin --mqtt-timeout 3000 --class-mode A >> ../evaluation/experiment2/all/class_a_${size}KiB.log
    echo "Finished sending random_${size}KiB.bin with class A"
    python3 application_server_claude.py switch-class --class C
done
for size in 1 2 5 10 50; do
    rm device_state_e21ca312cb69528a.json configuration_cache_e21ca312cb69528a.json
    python3 application_server_claude.py send-file --l 0.65 --file ../evaluation/experiment2/all/random_${size}KiB.bin --mqtt-timeout 3000 >> ../evaluation/experiment2/all/dynamic_${size}KiB.log
    echo "Finished sending random_${size}KiB.bin with Dynamic Method"
    python3 application_server_claude.py switch-class --class A
    python3 application_server_claude.py switch-class --class C
    python3 application_server_claude.py switch-class --class A
done