for pid in $(ps -ef | egrep 'aqi_sensor.py' | awk '{print $2}'); 
do
	echo "$pid";
	kill -9 $pid; 
done
