from numpy import *
import adaboost

'''
datMat,classLabels=adaboost.loadSimpData()
#print datMat,classLabels

#D = mat(ones((5,1))/5)
#print D

#bestStump,minError,bestClasEst=adaboost.buildStump(datMat,classLabels,D)
#print "bestStump=",bestStump," minError=",minError," bestClasEst=",bestClasEst

#classifierArray,aggCLassEst = adaboost.adaBoostTrainDS(datMat, classLabels, 30)
#print "classifierArray=",classifierArray
weakClassArr,aggClassEst = adaboost.adaBoostTrainDS(datMat, classLabels, 9)
print "weakClassArr=",weakClassArr,"aggClassEst=",aggClassEst

#print adaboost.adaClassify([0,0], classifierArray)
#print adaboost.adaClassify([1,0.9], classifierArray)

#print adaboost.adaClassify([[5,5],[0,0]], classifierArray)
'''


datArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
classifierArray,aggClassEst = adaboost.adaBoostTrainDS(datArr, labelArr, 10)

print "aggClassEst=",aggClassEst
#print classifierArray

#testArr, testLabelArr = adaboost.loadDataSet('horseColicTest2.txt')
#prediction10 = adaboost.adaClassify(testArr, classifierArray)
#print "prediction10=",prediction10
#errArr=mat(ones((67,1)))
#print errArr[prediction10 != mat(testLabelArr).T].sum()

print adaboost.plotROC(aggClassEst.T, labelArr)

