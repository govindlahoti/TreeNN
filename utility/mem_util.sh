#!/bin/bash

echo "TIME_STAMP,Memory Usage (MB)" | tee -a mem.csv
total="$(free -m | grep Mem | tr -s ' ' | cut -d ' ' -f 2)" 
while :
do
DATE=`date +"%H:%M:%S:%s%:z"`
echo -n "$DATE, " | tee -a mem.csv
var="$(top -b -n 1| grep -w $1 | tr -s ' ' | cut -d ' ' -f 11)"
echo "scale=3; ($var*$total/100)" | bc | tee -a mem.csv
sleep 1
done