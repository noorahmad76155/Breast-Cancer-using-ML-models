#Importing libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
import sklearn.model_selection as ms
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.random as nr
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix #added
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix #added

df = pd.read_csv('Brest Cancer Dataset.csv')
df.drop(['id'], axis = 1, inplace = True)
df.describe()

# 1 
concavity_mean = 1
for i in df['concavity_mean']:
  if i == 0:
    concavity_mean += 1
print(concavity_mean)
# 2 
concave_points_mean = 1
for i in df['concave points_mean']:
  if i == 0:
    concave_points_mean += 1
print(concave_points_mean)
# 3
symmetry_mean = 1
for i in df['symmetry_mean']:
  if i == 0:
    symmetry_mean += 1
print(symmetry_mean)
#only 14 zeros out of 569 data points is considerable

#Encoding Male and Female to 1 and 0
df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
df['diagnosis'].head(5)

#Assigning X,Y values
X = df.iloc[:, :-1].values
Y = df.iloc[:, 30].values
print("X: {}".format(X.shape))
print("Y: {}".format(Y.shape))

#Splitting X,Y into training & testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.175,random_state = 0)
print("X_train: {}".format(X_train.shape))
print("X_test: {}".format(X_test.shape))
print("Y_train: {}".format(Y_train.shape))
print("Y_test: {}".format(Y_test.shape))

#Building our baseline dummy classifier
from sklearn.dummy import DummyClassifier
clf = DummyClassifier()
clf.fit(X_train, Y_train)
#Predicting Results
y_pred = clf.predict(X_test)
#Calculating Resulta
print("CM: \n",confusion_matrix(Y_test, y_pred))
print("acc: {0}%".format(accuracy_score(Y_test, y_pred) * 100))

#AdaBoost classifier
nr.seed(1115)
ab_clf = AdaBoostClassifier()
ab_clf.fit(X_train, Y_train)
ab_clf_pred = ab_clf .predict(X_test)

#Model Evaluation
rmacc = accuracy_score(Y_test, ab_clf_pred)
print('Accuracy Score: ' + str(rmacc))
print('Precision Score: ' + str(precision_score(Y_test, ab_clf_pred)))
print('Recall Score: ' + str(recall_score(Y_test, ab_clf_pred)))
print('F1 Score: ' + str(f1_score(Y_test, ab_clf_pred)))
print('Classification Report: \n' + str(classification_report(Y_test, ab_clf_pred)))

#AdaBoost Model
import pickle
filename = 'Breast_Cancer.sav'
pickle.dump(ab_clf, open(filename, 'wb'))
