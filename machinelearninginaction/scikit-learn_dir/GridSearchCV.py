from sklearn import svm, grid_search, datasets
from pprint import pprint
from time import time

iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
#print clf.fit(iris.data, iris.target)

print("parameters:")
pprint(parameters)
t0 = time()
clf.fit(iris.data, iris.target)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % clf.best_score_)
print("Best parameters set:")
best_parameters = clf.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

