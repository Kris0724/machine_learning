#from svmutil import *

#y, x = svm_read_problem('/home/wangzengming/my_github/machine_learning/xgboost/demo/guide-python/dirty/samples_train.txt')
#t_y, _x = svm_read_problem('/home/wangzengming/my_github/machine_learning/xgboost/demo/guide-python/dirty/samples_test.txt')
#m = svm_train(y, x, '-c 4')
#p_label, p_acc, p_val = svm_predict(t_y, t_x, m)

#from liblinear import *
#prob = problem([1,-1], [{1:1, 3:1}, {1:-1,3:-1}])
#param = parameter('-c 4')
#>>> m = liblinear.train(prob, param) # m is a ctype pointer to a model
# Convert a Python-format instance to feature_nodearray, a ctypes structure
#>>> x0, max_idx = gen_feature_nodearray({1:1, 3:1})
#>>> label = liblinear.predict(m, x0)

from liblinearutil import *
# Read data in LIBSVM format
y, x = svm_read_problem('../heart_scale')
m = train(y[:200], x[:200], '-c 4')
p_label, p_acc, p_val = predict(y[200:], x[200:], m)


