# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1,2])], remainder="passthrough")
ct_country_gender = np.array(ct.fit_transform(X)[:, [1,2,3]], dtype=np.float)
X = np.hstack((ct_country_gender[:, :2], dataset.iloc[:, 3:4].values, ct_country_gender[:, [2]], dataset.iloc[:, 6:-1].values))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

# Adding the second hidden layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=6))

#adding the output layer
classifier.add(Dense(kernel_initializer="uniform", activation="sigmoid", units=1))#softmax for 2 or more independent results

#compilling the ANN
#adam is for stoichastric gradient for backward propagation
#if more than 2 dependent variables are there then we use categorical_crossentropy
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the ANN to the Training set
#epoch is the total restarting of all steps in ann
#batch size is after how many observations you want to add stoichastric gradient
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#accuracy=no. of correct obs./total obs.