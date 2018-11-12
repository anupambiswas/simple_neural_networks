Simple neural networks
======================

trivial_nn.py:
This is a simple python implementation of neural network for learning purposes.

It has straightforward implementations of the following neural network principles:
* Initialization of parameters
* Forward propagation to compute neural network output
* Back propagation to compute parameter gradients
* Gradient descent to compute newer values of parameters using a learning rate
* Cost computation
* Compute training and test accuracy

The test script (test.py) can be used to train a neural network. It creates training and test data based on a simple mathematical formula. 

How to run:
python3 test.py <train_set_size> <test_set_size> <number_of_training_iterations>

Example:
python3 test.py 10000 1000 5000000 # trains on a set of size 10000, tests a size of 1000, trains for 5000000 iterations 
