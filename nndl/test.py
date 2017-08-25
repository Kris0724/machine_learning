import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#print len(training_data[0][1])
#print training_data[0][0]
#exit(0)

import network
net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


