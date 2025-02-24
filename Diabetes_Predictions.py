import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
diabetes_dataset = pd.read_csv(r'C:\Users\Sasik\johith\MS_VS_CODE\InterPe\Diabetes.csv')
print(diabetes_dataset.head())
print(diabetes_dataset.describe())

#separate the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

#standardize the data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
print(X)
print(Y)

#split the data into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#train the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

#Accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data:', training_data_accuracy)

#making predictions on input data
input_data = (1, 85, 66, 29, 0, 26.6, 0.351, 31)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')