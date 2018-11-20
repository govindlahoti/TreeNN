for (( num=1; num<=36; num++ ))
do
	i=$(printf "%02d" $num)
	python3 aqi_sensor.py --sensor-id $i --source ~/TrainingData/0010$i_Data.csv -k $1 &
done