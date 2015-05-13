from numpy import *
import kNN

#group, labels = kNN.createDataSet()
#print group, labels

#tmp = kNN.classify0([0,0], group, labels, 3)
#print tmp

datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
#print datingDataMat
#print datingLabels

'''
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()
'''

#normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
#print normMat
#print ranges
#print minVals

#kNN.datingClassTest()

#kNN.classifyPerson()

#testVector = kNN.img2vector('testDigits/0_13.txt')
#print testVector[0, 0:31]

kNN.handwritingClassTest()
