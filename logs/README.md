# Logs

Each line corresponds to a log, which is a json dump.

Following is the format to interpret the log:

- node_id: ID of the node which reported the log
- type: Type of the log
- payload: Payload depends on the type of the log
  * CONN - A string indicating the information, else a dictionary in case of Network Cost (in bytes)
  * STAT - A dictionary having the following information depending on the node
    1. Parameter Server:
        - Merge ID
        - Pre Merge Accuracy
        - Post Merge Accuracy
    2. Worker:
        - Window ID
        - Run time : Time taken to process one window of data
        - Process time : CPU time
        - Memory Usage : Percentage of RAM used
        - Accuracy
  * DONE - Empty string
