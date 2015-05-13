#from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import numpy as np
n_samples, n_features = 10, 5
np.random.seed(0)
y = np.random.randn(n_samples)
print y
print
X = np.random.randn(n_samples, n_features)
print X
#clf = SVR(C=1.0, epsilon=0.2)
clf = KernelRidge(alpha=1.0)
clf.fit(X, y) 

print y[1]
print clf.predict(X[1])
