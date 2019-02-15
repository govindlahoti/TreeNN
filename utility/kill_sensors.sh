for pid in $(ps -ef | egrep '*sensor.py' | awk '{print $2}'); 
do
	echo "$pid";
	kill -9 $pid; 
done
