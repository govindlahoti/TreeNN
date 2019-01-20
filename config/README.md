# Config

![An sample configuration](../docs/images/config.png)

## Configuration fields
### Common fields for the hierarchy:
- Bandwidth between nodes pairwise
- Default bandwidth
- Time interval (in seconds) for a window
- Window limit for training
- Address of the kafka server
- Test directory

#### Machine info
- IP address: address of the host on which the node will run as a container
- Username of slave machine
- Password of slave machine

#### Docker specific fields:
- CPU resources to be allocated to the container
- Memory to be allocated to the container
- Docker image to be used
- Host test directory to be binded

#### Application arguments
#### Policy
Simple policy, Time-based policy, Window-based policy and Accuracy policy

### Fields for configuring nodes:
- Node id: Unique identification of a node
- Port: Port for starting the RPC server
- test_directory: Accuracies will be calculated against each test file present in the directory (Optional)
- Time interval for a window (Optional)
- Window limit (Optional)

## Link delays
Delays are sampled from a Gaussian distribution

1. tree.txt : Specify links for which delays are to be introduced
2. delays.yml : Output directed to this file

Run the following command:
```
python3 generate_link_delays.py -f tree.txt -m 1 -v 0.1
```

