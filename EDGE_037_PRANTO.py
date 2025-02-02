# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
path = r"C:\Users\User\PycharmProjects\pythonProject1\Social_Network_Ads.csv"
data = pd.read_csv(path)
print("The import data is \n",data)
# Insert some blank data and incorrect data types
data.loc[0:2, 'Age'] = np.nan  # Insert some blank data
if 'Gender' not in data.columns:
    data['Gender'] = [25, "Male", "Female", 30]  # Insert incorrect data types
# Select features and target
x = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values
# Handle missing data
x[:, 0] = np.nan_to_num(x[:, 0], nan=np.mean(x[~np.isnan(x[:, 0]), 0]))
print(" the processed data is \n",data)
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Apply MinMax scaling
scale1 = MinMaxScaler()
x_train = scale1.fit_transform(x_train)
x_test = scale1.transform(x_test)

# Train SVM model with linear kernel
cl1 = SVC(kernel='linear', random_state=0)
cl1.fit(x_train, y_train)

# Make predictions
y_predict = cl1.predict(x_test)

# Evaluate model performance
cm = confusion_matrix(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)

print("SVM Confusion Matrix:")
print(cm)
print(f"SVM Accuracy: {acc:.2f}")

# Train Random Forest model
cl2 = RandomForestClassifier(n_estimators=10, random_state=0)
cl2.fit(x_train, y_train)

# Make predictions
y_predict_rf = cl2.predict(x_test)

# Evaluate model performance
cm_rf = confusion_matrix(y_test, y_predict_rf)
acc_rf = accuracy_score(y_test, y_predict_rf)

print("Random Forest Confusion Matrix:")
print(cm_rf)
print(f"Random Forest Accuracy: {acc_rf:.2f}")

# Visualize SVM decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='winter', edgecolor='k', label='Training Data')
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_predict, cmap='autumn', marker='x', label='Predicted Test Data')
plt.title('SVM Decision Boundary Visualization')
plt.legend()
plt.show()