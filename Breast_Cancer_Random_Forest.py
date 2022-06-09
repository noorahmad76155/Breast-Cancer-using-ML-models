#importing required libraries: -
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix #added
from sklearn.dummy import DummyClassifier
import pickle

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.figure(figsize=(5,5))

#reading the data: -
df = pd.read_csv('Brest Cancer Dataset.csv')
df.shape
df.head(5)
df.drop(['id'], axis = 1, inplace = True)
df.describe()

#handling some lines in the dataset: -
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

#Setting X & Y values: -
X = df.iloc[:, :-1].values
Y = df.iloc[:, 30].values
print("X: {}".format(X.shape))
print("Y: {}".format(Y.shape))

#Splitting X&Y into training/ testing 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.175,random_state = 0)
print("X_train: {}".format(X_train.shape))
print("X_test: {}".format(X_test.shape))
print("Y_train: {}".format(Y_train.shape))
print("Y_test: {}".format(Y_test.shape))

#Building our baseline dummy classifier
clf = DummyClassifier()
clf.fit(X_train, Y_train)

#Predicting Results
y_pred = clf.predict(X_test)

#Calculating Resulta
print("CM: \n",confusion_matrix(Y_test, y_pred))
print("acc: {0}%".format(accuracy_score(Y_test, y_pred) * 100))

#Random Forest Classifier
st=dt.now()
randomforest = RandomForestClassifier(n_estimators = 100,random_state = 0)
randomforest.fit(X_train, Y_train)
print("Time taken to complete random search: ",dt.now()-st)
random_pred = randomforest.predict(X_test)

#Model Evaluation
rmacc = accuracy_score(Y_test, random_pred)
print('Accuracy Score: ' + str(rmacc))
print('Precision Score: ' + str(precision_score(Y_test, random_pred)))
print('Recall Score: ' + str(recall_score(Y_test, random_pred)))
print('F1 Score: ' + str(f1_score(Y_test, random_pred)))
print('Classification Report: \n' + str(classification_report(Y_test, random_pred)))

#Random Forest Model
filename = 'Breast_Cancer.sav'
pickle.dump(randomforest, open(filename, 'wb'))
