from svmutil import *

y, x = svm_read_problem('/home/wangzengming/my_github/machine_learning/xgboost/demo/guide-python/dirty/samples_train.txt')
t_y, _x = svm_read_problem('/home/wangzengming/my_github/machine_learning/xgboost/demo/guide-python/dirty/samples_test.txt')
m = svm_train(y, x, '-c 4')
p_label, p_acc, p_val = svm_predict(t_y, t_x, m)


