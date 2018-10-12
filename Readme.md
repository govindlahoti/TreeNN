# Simulator for IoT Analytics

## Simulator Design:
![Simulator Design](docs/images/Simulator%20Design.png)
## Dependencies for the Simulator:
[Kafka](https://hevodata.com/blog/how-to-install-kafka-on-ubuntu/)

[Kafka-python](https://pypi.org/project/kafka-python/)

## Instructions to use:

Clone the repository.

Specify the [configuration](config/network.yaml)

Run the following commands:
```
sudo /opt/kafka/bin/kafka-server-start.sh /opt/kafka/config/server.properties
python3 master.py --config config/network.yaml --log 1
cd sensor && ./start_sensors.sh
```

## Directory Structure:

- [application](application): Contains the code of the application simulated
- [config](config): Contains yaml configurations for different experiments
- [logs](logs): Logs stored for each node
- [node](node): Code for simulating edge nodes
- [sensor](sensor): Code for simulating sensors
- [utility](utility): Contains utility bash scripts and python helpers


