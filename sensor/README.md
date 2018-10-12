# Sensor

Contains code for Simulating sensor

### How to run:
```
python3 aqi_sensor.py --sensor_id 1 --topic 2 --source Cluster6_Data.csv --data_rate 0.1
```

### Mechanism:
Currently the sensor reads data from a file and dumps into Kafka, essentially acting as a Kafka server
- sensor_id (Optional): Sensor ID
- topic: Topic to be published on, Node id for the purpose of the Simulator.
- source: Data file from which sensor will dump the data into Kafka
- data_rate (Optional): Time interval between sending two data measurements, default value is picked up from [const.py](../utility/const.py)