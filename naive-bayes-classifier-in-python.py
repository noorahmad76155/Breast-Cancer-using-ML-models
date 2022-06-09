#Importing Libraries:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix #added
import pickle

import warnings
warnings.filterwarnings('ignore')

#Reading the dataset
df = pd.read_csv('Brest Cancer Dataset.csv')
df.shape
df.drop(['id'], axis = 1, inplace = True)
df.describe()

df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
df['diagnosis'].head(5)

#Setting X and Y values :
X = df.iloc[:, :-1].values
Y = df.iloc[:, 30].values
print("X: {}".format(X.shape))
print("Y: {}".format(Y.shape))

# split X and y into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# check the shape of X_train and X_test
X_train.shape, X_test.shape

# train a Gaussian Naive Bayes classifier on the training set
# instantiate the model
gnb = GaussianNB()

# fit the model
gnb.fit(X_train, Y_train)

y_pred = gnb.predict(X_test)
y_pred

#Model Evaluation
rmacc = accuracy_score(Y_test, y_pred)
print('Accuracy Score: ' + str(rmacc))
print('Precision Score: ' + str(precision_score(Y_test, y_pred)))
print('Recall Score: ' + str(recall_score(Y_test, y_pred)))
print('F1 Score: ' + str(f1_score(Y_test, y_pred)))
#print('Classification Report: \n' + str(classification_report(Y_test, y_pred)))

#NaiveBayes Model
filename = 'Breast_Cancer.sav'
pickle.dump( gnb, open(filename, 'wb'))

# Now, we will compare the train-set and test-set accuracy to check for overfitting.
Y_pred_train = gnb.predict(X_train)
Y_pred_train
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(Y_train, Y_pred_train)))

# print the scores on training and test set
print('Training set score: {:.4f}'.format(gnb.score(X_train, Y_train)))
print('Test set score: {:.4f}'.format(gnb.score(X_test, Y_test)))

# check null accuracy score
null_accuracy = (7407/(7407+2362))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

# visualize confusion matrix with seaborn heatmap
print("CM: \n",confusion_matrix(Y_test, y_pred))

# We can print a classification report as follows:-
print(classification_report(Y_test, y_pred))

