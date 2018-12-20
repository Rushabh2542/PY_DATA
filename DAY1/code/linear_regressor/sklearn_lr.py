import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd

# Load CSV and columns for Housing.csv
df = pd.read_csv('Housing.csv')
X = df['plotsize']
bedrooms = df['bedrooms']
Y = df['price']

# Get numpy array
X = np.array(X)
bedrooms = np.array(bedrooms)
Y = np.array(Y)

# Reshape data into (row, 1)
X = X.reshape(len(X), 1)
bedrooms = bedrooms.reshape(len(bedrooms), 1)
Y = Y.reshape(len(Y), 1)

# Join data column into with axis=1
X = np.append(X, bedrooms, axis=1)

# Split dataset in train/test
X_train = X[:X.shape[0]*80//100, :]
Y_train = Y[:Y.shape[0]*80//100, :]

X_test = X[X.shape[0]*80//100:, :]
Y_test = Y[Y.shape[0]*80//100:, :]

# Get sklearn model = linear_model.LinearRegression()
model = linear_model.LinearRegression()

# Start training
print("Start Training...")
model.fit(X_train, Y_train)

# Predict
y_pred = model.predict(X_test)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], Y_test[:,0])
ax.plot(X_test[:, 0], X_test[:, 1], y_pred[:,0], color='red',linewidth=3)
plt.show()



























"""
# Load CSV and columns
df = pd.read_csv('Housing.csv')

X = df['plotsize']
Y = df['price']
bedrooms = df['bedrooms']

X = np.array(X)
Y = np.array(Y)
bedrooms = np.array(bedrooms)

X = X.reshape((len(X), 1))
Y = Y.reshape((len(Y), 1))
bedrooms = bedrooms.reshape((len(bedrooms), 1))

X =np.append(X, bedrooms, axis = 1)

# Split the data into training/testing sets
X_train = X[:400, :]
X_test = X[400:, :]

# Split the targets into training/testing sets
Y_train = Y[:400, :]
Y_test = Y[400:, :]

# Plot outputs
model = linear_model.LinearRegression()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y[:,0])

# Train the model using the training sets
print('Start training')
model = model.fit(X_train, Y_train)

#testing part
Y_pred = model.predict(X_test)

ax.plot(X_test[:, 0], X_test[:, 1], Y_pred[:,0],color='red',linewidth=3)

# Plot outputs
plt.show()
"""