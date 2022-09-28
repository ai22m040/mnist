# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 14:08:15 2022

@author: kloimstg
"""

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier




# splits the data in train and test, builds the model and then predicts
# for a randomly drawn sample out of test

print("Importing dataset...")

 # import MNIST Dataset from sklearn
#my_data = load_digits()
mnist = fetch_openml('mnist_784')


# store feature matrix in "X"
X = mnist.data

# store target vector in "y"
y = mnist.target

      
# Split data into 80% train and 20% test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Use KNN
my_neighbors = 8
my_weights = 'uniform'
clf = KNeighborsClassifier(n_neighbors=my_neighbors, weights=my_weights)
clf.fit(X_train, y_train)


# Make predictions
y_pred = clf.predict(X_test)

# Check the accuracy of the model
acc_score = metrics.accuracy_score(y_test, y_pred)
print ("KLN accuracy score: ", acc_score)

# Save the model
filename = 'mnist_KNN_model.sav'
pickle.dump(clf, open(filename, 'wb'))

print (f"Model saved as {filename}")
        


