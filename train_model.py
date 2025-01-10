import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("insurance3r2.csv")

# Drop missing values and unwanted columns
data = data.dropna()
data = data.drop('region', axis=1, errors='ignore')

# Convert categorical variables to numeric
data['sex'] = data['sex'].map({'male': 1, 'female': 0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})

# Split the data into features and target
X = data.drop('insuranceclaim', axis=1)
y = data['insuranceclaim']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
rf = RandomForestClassifier()

# Train the model
rf.fit(X_train, y_train)

# Save the trained model
pickle.dump(rf, open('model.pkl', 'wb'))

print("Model saved successfully as 'model.pkl'")
