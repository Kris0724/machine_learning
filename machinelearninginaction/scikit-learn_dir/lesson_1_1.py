from numpy import *
from sklearn import linear_model


def readfile1(filename):
	#f = open('iris.data.all')
	f = open(filename)
	line = f.readline()
	num_feat = len(line.split(',')) - 1
	data_mat = []
	label_mat = []
	#f2 = open('iris.data.all')
	f2 = open(filename)
	for line in f2.readlines():
		if line.strip() == '':
			line = f.readline()
			continue
		line = line.strip('\n')
		arr = line.split(',')
		f_arr = []
		for i in range(num_feat):
			f_arr.append(float(arr[i]))
		data_mat.append(f_arr)
		label_mat.append(float(arr[-1]))		
	return data_mat,label_mat


data_mat, label_mat = readfile1('./bak/iris.data.svm.train')

###  LinearRegression
'''
clf = linear_model.LinearRegression()
clf.fit(data_mat, label_mat)
print clf.predict(data_mat)
print clf.score(data_mat, label_mat)
'''

'''
clf = linear_model.Ridge (alpha = .5)
clf.fit(data_mat, label_mat)
print clf.predict(data_mat)
print clf.score(data_mat, label_mat)
'''

'''
clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
clf.fit(data_mat, label_mat)
print clf.predict(data_mat)
print clf.score(data_mat, label_mat)
print clf.alpha_
'''


#clf = linear_model.Lasso(alpha=0.1)
clf = linear_model.LassoCV(alphas=[0.1, 1.0, 10.0])
clf.fit(data_mat, label_mat)
print clf.predict(data_mat)
#print clf.score(data_mat, label_mat)
#print clf.coef_
#print clf.intercept_
print clf.alpha_











