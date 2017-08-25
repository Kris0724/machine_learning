import mnist_loader
import numpy as np
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print training_data[0]
exit(0)

#training_data = [ [ [0,0], [1,1], [0.5,0.5], [-0.5,-0.5], [-1, -1], [-0.8,-0.7] ],[ [1, 0], [1,0], [1,0], [0,1], [0,1], [0,1] ] ]
#training_data = np.array([[[[0],[0]],[[1],[1]],[[0.5],[0.5]],[[-0.5],[-0.5]],[[-1], [-1]],[[-0.8],[-0.7]]],[[[1], [0]],[[1],[0]],[[1],[0]],[[0],[1]],[[0],[1]],[[0],[1]]]])
#test_data = [ [ [0.1,0.2], [0.9,1], [-0.8, -0.9], [-0.7,-0.7] ],[ [1,0], [1,0], [0,1], [0,1] ] ]

#test_data = np.array([[[[0.7],[0.7]],[[0.8],[0.8]],[[-0.7],[-0.9]],[[-0.8],[-0.8]]],[[[1],[0]],[[1],[0]],[[0],[1]],[[0],[1]]]])


#print training_data
#print test_data
#print len(training_data[0][1])
#print training_data[0][0]
#exit(0)

import network
net = network.Network([784, 30, 10])
#net = network.Network([2, 5, 2])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
#net.SGD(training_data, 30, 1, 3.0, test_data=test_data)


