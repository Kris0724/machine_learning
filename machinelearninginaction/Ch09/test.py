from numpy import *

import regTrees

#testMat = mat(eye(4))
#print testMat
#mat0, mat1 = regTrees.binSplitDataSet(testMat, 2, 2.5)
#print mat0
#print 
#print mat1

#myDat = regTrees.loadDataSet('ex00.txt')
#myDat = regTrees.loadDataSet('tt3.txt')
#myDat = regTrees.loadDataSet('ex0.txt')
#print myDat
#print
#myMat = mat(myDat)
#print regTrees.createTree(myMat)
#print regTrees.createTree(myMat, ops=(0,1))



#myDat1 = regTrees.loadDataSet('ex0.txt')
#myMat1 = mat(myDat1)
#print regTrees.createTree(myMat1)


#myDat2 = regTrees.loadDataSet('ex2.txt')
#myDat2 = regTrees.loadDataSet('ex2.txt')
#myMat2 = mat(myDat2, )
#print myDat2
#print regTrees.createTree(myMat2, ops=(10000,4))



#model tree
#myMat2 = mat(regTrees.loadDataSet('exp2.txt'))
#print regTrees.createTree(myMat2, regTrees.modelLeaf, regTrees.modelErr, (1,10))


trainMat = mat(regTrees.loadDataSet('bikeSpeedVsIq_train.txt'))
testMat = mat(regTrees.loadDataSet('bikeSpeedVsIq_test.txt'))
myTree = regTrees.createTree(trainMat, ops=(1,20))
#print myTree
#___REGRES TREE___
yHat = regTrees.createForeCast(myTree, testMat[:,0])
#print yHat
print corrcoef(yHat, testMat[:,1], rowvar=0)[0,1]
#___MODEL TREE___
myTree = regTrees.createTree(trainMat, regTrees.modelLeaf, regTrees.modelErr, (1,20))
yHat = regTrees.createForeCast(myTree, testMat[:,0], regTrees.modelTreeEval)
print corrcoef(yHat, testMat[:,1], rowvar=0)[0,1]
#___STAND REGRES___
ws,X,Y = regTrees.linearSolve(trainMat)
print "ws=",ws
for i in range(shape(testMat)[0]):
    yHat[i] = testMat[i,0]*ws[1,0]+ws[0,0]
print corrcoef(yHat, testMat[:,1], rowvar=0)[0,1]




