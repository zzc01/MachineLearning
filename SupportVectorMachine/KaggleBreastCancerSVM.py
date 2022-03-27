###########################################################################
#
# References: 
# [1] https://www.kaggle.com/uciml/breast-cancer-wisconsin-data 
# [2] Sandex, Machine Learning with Python, https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v 
# [3] CVXOPT tutorial  https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
#
###########################################################################
 

from sre_parse import expand_template
import numpy as np
from sklearn import preprocessing, neighbors, svm, model_selection 
# from sklearn import cross_validation 
import pandas as pd 

# df = pd.read_csv('/archive/data.csv')
df = pd.read_csv('data.csv')
df.replace('?', -9999, inplace=True)


df.drop(['id'], 1, inplace=True)

print("What is this? {}".format(df.columns.str.contains('unamed', case=False)) )
#df.drop(, axis=1, inplace=True)

df.drop(df.columns[df.columns.str.contains('unnamed', case = False)], axis = 1, inplace = True)


X = np.array(df.drop(['diagnosis'], 1))
y = np.array(df['diagnosis'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
#clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print("Accuracy on training data : {}".format(accuracy) )


example_mesures = np.array([
    [17.99,	10.38,	122.8,	1001,	0.1184,	0.2776,	0.3001,	0.1471,	0.2419,	0.07871,	1.095,	0.9053,	8.589,	153.4,	0.006399,	0.04904,	0.05373,	0.01587,	0.03003,	0.006193,	25.38,	17.33,	184.6,	2019,	0.1622,	0.6656,	0.7119,	0.2654,	0.4601,	0.1189],
    [17.99,	10.38,	122.8,	1001,	0.1184,	0.2776,	0.3001,	0.1471,	0.2419,	0.07871,	1.095,	0.9053,	8.589,	153.4,	0.006399,	0.04904,	0.05373,	0.01587,	0.03003,	0.006193,	25.38,	17.33,	184.6,	2019,	0.1622,	0.6656,	0.7119,	0.2654,	0.4601,	0.1189],
])

# print(example_mesures.size)
# print(type(example_mesures))
# print(example_mesures)

example_mesures = example_mesures.reshape(len(example_mesures), -1)

# print(example_mesures.size)
# print(type(example_mesures))
# print(example_mesures)

prediction = clf.predict(example_mesures)
print(prediction)

