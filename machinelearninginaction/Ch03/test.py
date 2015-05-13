import trees

myData, labels = trees.createDataSet2()
#print myData,labels

#print trees.calcShannonEnt(myData)

#print trees.chooseBestFeatureToSplit(myData)

myTree = trees.createTree(myData, labels)
print myTree


import treePlotter
treePlotter.createPlot(myTree)
'''
import treePlotter
#treePlotter.createPlot()

#myTree = treePlotter.retrieveTree(1)
myTree = treePlotter.retrieveTree(0)
print myTree
#print treePlotter.getNumLeafs(myTree)
#print treePlotter.getTreeDepth(myTree)

#treePlotter.createPlot(myTree)

myData, labels = trees.createDataSet()
print labels

print trees.classify(myTree, labels, [1,0])
print trees.classify(myTree, labels, [1,1])
'''

'''
import treePlotter
fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
#print lenses
lensesLabels=['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses, lensesLabels)
print  lensesTree

treePlotter.createPlot(lensesTree)
'''


