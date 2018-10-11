for pid in $(ps -ef | egrep 'python3 master.py' | awk '{print $2}'); 
do
	echo "$pid";
	kill -9 $pid; 
done
