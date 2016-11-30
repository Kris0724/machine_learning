from numpy import *
#import sys
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

def readfile1(filename):
	#f = open('iris.data.all')
	f = open(filename)
	line = f.readline()
	num_feat = len(line.split('\t')) - 1
	data_mat = []
	label_mat = []
	#f2 = open('iris.data.all')
	f2 = open(filename)
	for line in f2.readlines():
		if line.strip() == '':
			line = f.readline()
			continue
		line = line.strip('\n')
		arr = line.split('\t')
		f_arr = []
		for i in range(num_feat):
			f_arr.append(float(arr[i]))
		data_mat.append(f_arr)
		label_mat.append(int(float(arr[-1])))		
	return data_mat,label_mat

#data_mat, label_mat = readfile1('./bak/iris.data.all')
data_mat, label_mat = readfile1('./horseColicTraining.txt')
data_mat2, label_mat2 = readfile1('./horseColicTest.txt')
#print data_mat
#print label_mat
#exit(0)

### GBDT
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3).fit(data_mat, label_mat)
#print clf.score(data_mat2, label_mat2)

### RF
rfc = ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_features="auto", min_samples_split=1).fit(data_mat, label_mat)
print rfc.score(data_mat2, label_mat2)

###  ADABOOST
'''
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(data_mat, label_mat)
print clf.score(data_mat, label_mat)
predict_labels = clf.predict(data_mat)
print predict_labels
print clf.score(data_mat2, label_mat2)
predict_labels = clf.predict(data_mat2)
print predict_labels

#l = len(predict_labels)
#err_count = 0
#for i in range(l):
#	if predict_labels[i] != label_mat2[i]:
#		err_count += 1
#print err_count
#print l
#print float(l - err_count)/l
scores = cross_val_score(clf, data_mat, label_mat)
print scores
'''

'''
clf = AdaBoostRegressor()
clf.fit(data_mat, label_mat)
print clf.score(data_mat, label_mat)
predict_labels = clf.predict(data_mat)
print predict_labels
'''

'''
clf = AdaBoostRegressor(n_estimators=100)
clf.fit(data_mat, label_mat)
print clf.score(data_mat, label_mat)
print clf.predict(data_mat)

print clf.score(data_mat2, label_mat2)
print clf.predict(data_mat2)
#error,can not do like this
scores = cross_val_score(clf, data_mat, label_mat)
print scores
'''

###  LOGISTIC REGRESSION
'''
clf = LogisticRegression(C=0.01)
clf.fit(data_mat, label_mat)
predict_labels = clf.predict(data_mat)
#print predict_labels
predict_labels = clf.predict(data_mat2)
#print predict_labels
print clf.score(data_mat, label_mat)
print clf.score(data_mat2, label_mat2)
scores = cross_val_score(clf, data_mat, label_mat)
print scores
'''

###  NAIVE BAYES
'''
clf = GaussianNB()
clf.fit(data_mat, label_mat)
#print(clf.predict(data_mat))
print clf.score(data_mat, label_mat)
scores = cross_val_score(clf, data_mat, label_mat)
print scores
'''

###   DECISION TREE
'''
clf = DecisionTreeClassifier(random_state=0)
clf.fit(data_mat, label_mat)
print(clf.predict(data_mat))
print clf.score(data_mat, label_mat)

print(clf.predict(data_mat2))
print clf.score(data_mat2, label_mat2)

scores = cross_val_score(clf, data_mat, label_mat)
print scores
'''

'''
clf = DecisionTreeRegressor(random_state=0)
clf.fit(data_mat, label_mat)
print(clf.predict(data_mat))
print clf.score(data_mat, label_mat)

print(clf.predict(data_mat2))
print clf.score(data_mat2, label_mat2)
scores = cross_val_score(clf, data_mat, label_mat)
print scores
'''

###   K NEAREST NEIGHBOUR
'''
clf = NearestNeighbors()
print clf.fit(data_mat)
#print clf.kneighbors(data_mat, n_neighbors=1, return_distance=False) 
print clf.kneighbors(data_mat, n_neighbors=1) 
'''

### LINEAR REGRESSION
'''
clf = linear_model.LinearRegression()
clf.fit(data_mat, label_mat)
print clf.predict(data_mat)
print clf.predict(data_mat2)
#print clf.score(data_mat, label_mat)
#error, can not do like this
scores = cross_val_score(clf, data_mat, label_mat)
print scores
'''

###  SVM
'''
clf = SVC()
clf.fit(data_mat, label_mat)
print clf.predict(data_mat)
print clf.predict(data_mat2)
print clf.score(data_mat, label_mat)
print clf.score(data_mat2, label_mat2)
scores = cross_val_score(clf, data_mat, label_mat)
print scores
'''

'''
clf = SVC(kernel='rbf')
#clf = SVC(kernel='linear')
#clf = NuSVC()
#clf = LinearSVC()
#clf = svm.SVR()
clf.fit(data_mat, label_mat)
print clf.predict(data_mat)
print clf.predict(data_mat2)
print clf.score(data_mat, label_mat)
print clf.score(data_mat2, label_mat2)
#error, can not do like this
scores = cross_val_score(clf, data_mat, label_mat)
print scores
'''






#---------------------------------------------------

'''
train_data, train_target = readfile1('./bak/iris.data.all')
lr = linear_model.LogisticRegression()
lr_scores = cross_validation.cross_val_score(lr, train_data, train_target, cv=5)
print("logistic regression accuracy:")
print(lr_scores)

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=5)
clf_scores = cross_validation.cross_val_score(clf, train_data, train_target, cv=5)
print("decision tree accuracy:")
print(clf_scores)

rfc = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=3, max_features=0.5, min_samples_split=5)
rfc_scores = cross_validation.cross_val_score(rfc, train_data, train_target, cv=5)
print("random forest accuracy:")
print(rfc_scores)

etc = ensemble.ExtraTreesClassifier(criterion='entropy', n_estimators=3, max_features=0.6, min_samples_split=5)
etc_scores = cross_validation.cross_val_score(etc, train_data, train_target, cv=5)
print("extra trees accuracy:")
print(etc_scores)

gbc = ensemble.GradientBoostingClassifier()
gbc_scores = cross_validation.cross_val_score(gbc, train_data, train_target, cv=5)
print("gradient boosting accuracy:")
print(gbc_scores)

svc = svm.SVC()
svc_scores = cross_validation.cross_val_score(svc, train_data, train_target, cv=5)
print("svm classifier accuracy:")
print(svc_scores)
'''

'''
from sklearn import svm
#clf = svm.LinearSVC()
svc = svm.SVC(kernel='rbf')
#clf.fit(data, label)
print svc.fit(data, label)
err_count = 0
m,n = shape(data)
for i in range(m):
	#predict = clf.predict(data[i])
	predict = svc.predict(data[i])
	if predict != label[i]:
		err_count +=1
print "train err=", err_count
print "train sum=", m
m,n = shape(data2)
err_count = 0
for i in range(m):
	#predict = clf.predict(data[i])
	predict = svc.predict(data[i])
	if predict != label2[i]:
		err_count += 1
print "test err=", err_count
print "test sum=", m
'''





