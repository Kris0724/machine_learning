"""
================================================================
Plot the decision surface of a decision tree on the iris dataset
================================================================

Plot the decision surface of a decision tree trained on pairs
of features of the iris dataset.

See :ref:`decision tree <tree>` for more information on the estimator.

For each pair of iris features, the decision tree learns decision
boundaries made of combinations of simple thresholding rules inferred from
the training samples.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
plot_colors = "bry"
plot_step = 0.02

# Load data
iris = load_iris()

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    #print pairidx,pair
    #continue;
    #exit(0)

    # We only take the two corresponding features
    X = iris.data[:, pair]
    #print X
    y = iris.target
    #print y
    #exit(0)

    # Shuffle
    idx = np.arange(X.shape[0])
    #print idx
    #exit(0)
    np.random.seed(13)
    np.random.shuffle(idx)
    #print idx
    #exit(0)
    X = X[idx]
    y = y[idx]

    # Standardize
    mean = X.mean(axis=0)
    #print mean
    std = X.std(axis=0)
    #print std
    X = (X - mean) / std
    #print X
    #exit(0)

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    #print np.c_[xx.ravel(), yy.ravel()]
    #exit(0)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #print Z
    #print
    Z = Z.reshape(xx.shape)
    #print xx.shape
    #print
    #exit(0)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    plt.axis("tight")

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.Paired)

    plt.axis("tight")

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend()
plt.show()
