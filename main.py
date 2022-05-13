# Import libraries
import pandas as pd
import sklearn
from sklearn import linear_model


# Read the CSV file
data = pd.read_csv('student-mat.csv', sep=';', header=0)

#print(data.head())

# choose input and target variable
X = data[['G1', 'G2', 'studytime', 'failures', 'absences']]
y = data['G3']


# split data into train and test
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


# let's create a model
model = linear_model.LinearRegression()

# train the model
model.fit(X_train, y_train)

# test the model accuracy
acc = model.score(X_test, y_test)
print(acc)


# printout the COEFFICIENT & INTERCEPT
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)


# prediction
predictions = model.predict(X_test)

for x in range(len(predictions)):
    print(predictions[x], X_test[x], y_test[x])
