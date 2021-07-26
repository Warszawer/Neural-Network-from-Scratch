import mnist_loader
trainig_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network_matrix
net = network_matrix.Network([784, 30, 10])
net.SGD(trainig_data, 30, 10, 3.0, test_data=test_data)