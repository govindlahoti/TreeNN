# Simulator for IoT Analytics

## Simulator Design:
![Simulator Design](docs/images/Simulator%20Design.png)

## Instructions to use:

Clone the repository.

Specify the [configuration](config/docker/test.yaml)

Run the following commands:
```
## Building the docker image
cd docker; ./build.sh; cd ..

## Triggering containers
python3 master.py --expname test --config config/network.yaml --ip localhost 
```

## Directory Structure:

- [app](app): Code of the application to be simulated
- [config](config): Yaml configurations for different experiments
- [docker](docker): Docker build files
- [docs](docs): Documentation, references and reports
- [evaluation](evaluation): Scripts for analyzing logs
- [logs](logs): Logs obtained from experiments
- [node](node): Code for simulating edge nodes
- [policy](policy): Code for model exchange policies
- [sensor](sensor): Code for simulating sensors
- [utility](utility): Contains utility bash scripts and python helper functions


