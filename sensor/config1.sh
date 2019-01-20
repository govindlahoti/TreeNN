# for (( num=1; num<=36; num++ ))
# do
# 	i=$(printf "%02d" $num)
# 	python3 aqi_sensor.py --sensor-id $num --source ~/TrainingData/0010${i}_Data.csv -k $1 &
# done

python3 aqi_sensor.py --sensor-id 1 --source ~/TrainingData/001036_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 2 --source ~/TrainingData/001035_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 3 --source ~/TrainingData/001017_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 4 --source ~/TrainingData/001034_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 5 --source ~/TrainingData/001004_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 6 --source ~/TrainingData/001003_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 7 --source ~/TrainingData/001010_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 8 --source ~/TrainingData/001019_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 9 --source ~/TrainingData/001025_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 10 --source ~/TrainingData/001001_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 11 --source ~/TrainingData/001007_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 12 --source ~/TrainingData/001033_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 13 --source ~/TrainingData/001024_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 14 --source ~/TrainingData/001023_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 15 --source ~/TrainingData/001026_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 16 --source ~/TrainingData/001029_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 17 --source ~/TrainingData/001027_Data.csv -d $1 -k $2 &
python3 aqi_sensor.py --sensor-id 18 --source ~/TrainingData/001028_Data.csv -d $1 -k $2 &
