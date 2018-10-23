#!/bin/bash
echo "writing to cpu.csv"
echo "TIME_STAMP, Usage%" | tee -a cpu.csv
while :
do
DATE=`date +"%H:%M:%S:%s%:z"`
echo -n "$DATE, " | tee -a cpu.csv
top -b -n 1| grep -w $1 | tr -s ' ' | cut -d ' ' -f 10 | tee -a cpu.csv
sleep 1
done