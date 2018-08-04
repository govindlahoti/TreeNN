for pid in $(ps -ef | grep "python slave.py" | awk '{print $2}'); 
	do kill -9 $pid; 
done