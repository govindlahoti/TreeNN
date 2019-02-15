for (( num=1; num<=36; num++ ))
do
    if [[ $(($num % 2)) -eq 1 ]]
    then
        i=$(printf "%02d" $num)
        python3 aqi_sensor.py --sensor-id $num --source ../../../TrainingData/0010${i}_Data.csv -k $1 -d 0.1 &
    fi
done
