import kNN
#group,labels=kNN.createDataSet()
#kNN.classify0([0,0], group, labels, 3)

datingDateMat,datingLabels=kNN.file2matrix('datingTestSet2.txt')

import matplotlib
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(datingDateMat[:,1], datingDateMat[:,2])
plt.show()

