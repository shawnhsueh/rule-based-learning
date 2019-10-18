 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

training_data = pd.read_csv('C:/Users/Chung/rule-based-learning/datasets/humanSP1/humanSP1_train.csv', sep= ',', header=None)
X = training_data.values[:,0]
Y = training_data.values[:, 1:2]

updated_X = []

for line in X:
	tmp= []
	for character in line:
		if character == 'A':
			tmp.append("0001")
		elif character == 'C':
			tmp.append("0010")
		elif character == 'G':
			tmp.append("0100")
		elif character == 'T':
			tmp.append("1000")
	updated_X.append(tmp)
#    
#for line in X:
#    tmp= np.zeros((4, 14))
#    for i in range(len(line)):
#        if  line[i] == 'A':
#            tmp[0][i] = 1
#        elif line[i] == 'C':
#            tmp[1][i] = 1
#        elif line[i] == 'G':
#            tmp[2][i] = 1
#        elif line[i] == 'T':
#            tmp[3][i] = 1
#    tmp = tmp.flatten()
#    updated_X.append(tmp)

X_train, X_test, y_train, y_test = train_test_split(updated_X, Y, test_size = 0.3, random_state = 100)

tfbs_classifier = tree.DecisionTreeClassifier()

scores = np.zeros(100)
for i in range(100):
    tfbs_classifier = tfbs_classifier.fit(X_train, y_train)
    y_pred = tfbs_classifier.predict(X_test)
    scores[i] = f1_score(y_test, y_pred, average='macro') 
print(np.mean(scores), np.std(scores))
