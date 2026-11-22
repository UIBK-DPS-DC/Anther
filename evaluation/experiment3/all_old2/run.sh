#!/usr/bin/env bash

# This script is used to execute multiple downlinks in squence.
# The structure is to run downlinks with Class A and Class C in combination with different amounts of data.
# The aim of this is to compare the measured energy and time consumption with the predicted once.
# Additionally the script uses class switching to easily mark timestamps in the power consumption data. Needs to be manually cut.
# 1 2 5 10 50
# Generate nice file sizes, with random content. 
cd ~/deltafuota/networkserver
python3 application_server.py set-datarate --datarate 5
python3 application_server.py switch-class --class A
python3 application_server.py switch-class --class C
python3 application_server.py switch-class --class A
python3 application_server.py set-interval --seconds 60
for size in 1 2 5 10 50; do
    dd if=/dev/urandom \
       of="../evaluation/experiment3/all/random_${size}KiB.bin" \
       bs=1K count="$size" \
       status=none
done
# for size in 1 2 5 10 50; do
#     python3 application_server.py send-file --file ../evaluation/experiment3/all/random_${size}KiB.bin --adr-override --mqtt-timeout 3000 --class-mode C >> ../evaluation/experiment3/all/class_c_${size}KiB.log
#     echo "Finished sending random_${size}KiB.bin with class C"
#     python3 application_server.py set-datarate --datarate 5
#     python3 application_server.py switch-class --class A
#     python3 application_server.py switch-class --class C
#     python3 application_server.py switch-class --class A
# done
# for size in 1 2 5 10 50; do
#     python3 application_server.py send-file --file ../evaluation/experiment3/all/random_${size}KiB.bin --adr-override --mqtt-timeout 3000 --class-mode A >> ../evaluation/experiment3/all/class_a_${size}KiB.log
#     echo "Finished sending random_${size}KiB.bin with class A"
#     python3 application_server.py set-datarate --datarate 5
#     python3 application_server.py switch-class --class A
#     python3 application_server.py switch-class --class C
#     python3 application_server.py switch-class --class A
# done

for lambda in 0.4 0.65; do
    for size in 1 2 5 10 50; do
        python3 application_server.py send-file --file ../evaluation/experiment3/all/random_${size}KiB.bin --l ${lambda} --adr-override --mqtt-timeout 3000 >> ../evaluation/experiment3/all/lambda_${lambda}_new/dynamic_${size}KiB.log
        echo "Finished sending random_${size}KiB.bin with Dynamic Method"
        python3 application_server.py set-datarate --datarate 5
        python3 application_server.py switch-class --class A
        python3 application_server.py switch-class --class C
        python3 application_server.py switch-class --class A
    done
done