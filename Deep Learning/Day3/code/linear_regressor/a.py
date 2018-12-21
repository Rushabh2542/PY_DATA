import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd

# Load CSV and columns
df = pd.read_csv('Housing.csv')

X = df['plotsize']
Y = df['price']
bedrooms = df['bedrooms']
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