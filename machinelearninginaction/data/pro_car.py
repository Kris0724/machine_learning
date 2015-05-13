from numpy import *
import sys

sys.path.append('../Ch03/')
import trees
import treePlotter

f = open('./car.data')
cars = [inst.strip().split(',') for inst in f.readlines()]
#m,n = shape(cars)
#print m,n

#print cars
cars_labels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
cars_labels2 = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

test_set = []
test_label = []	
for i in range(100):
	rand_index = int(random.uniform(0,len(cars)))
	test_set.append(cars[rand_index][0:-1])
	test_label.append(cars[rand_index][-1])
	del(cars[rand_index])
#print test_set
#print
#print test_label
#exit(0)

print cars_labels

#m,n = shape(cars)
#print m,n

#m,n = shape(test_set)
#print m,n
cars_tree = trees.createTree(cars, cars_labels)
#print cars_tree

m,n = shape(test_set)

#print cars_labels
#exit(0)
#print cars_labels2
#exit(0)

err_count = 0
for i in range(m):
	ret = trees.classify(cars_tree, cars_labels2, test_set[i])
	if ret != test_label[i]:
		err_count +=1
print "err=", err_count
print "sum=", m

#treePlotter.createPlot(cars_tree)



