# Node

node.py : An abstract class for simulating an edge device

## Parameter Server:

Parameter server spawns two threads - 
- To consume the gradients being obtained from the child node
  1. Monitors the queue of acquired gradients from the child nodes. 
  2. Pulls model from parent
  3. Logs the accuracy of the model before merging gradients
  4. Modifies the node's model using those gradients.
  5. Logs the accuracy of the model after merging gradients (merging methods can be multiple)
  6. Pushes new model to the parent
 
- To run the RPC server

## Worker:

Worker spawns two threads - 
- To run the training thread. For each window of data:
  1. Pulls model from parent 
  2. Runs training algorithm as defined by the application
  3. Log Statistics for the epoch:
      * Window ID
      * Runtime
      * Process time
      * Memory usage recorded at the end of the epoch 
      * Accuracy of the model
  4. Push model to the parent 
- To run the RPC server

