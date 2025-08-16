import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset
df = pd.read_csv(r"C:\Users\21311\Downloads\onlinefraud.csv")

print(df.dtypes)

def reduce_cardinality(column, threshold=0.01):
    value_counts = column.value_counts(normalize=True)
    to_replace = value_counts[value_counts < threshold].index
    column = column.apply(lambda x: 'Other' if x in to_replace else x)
    return column

# List of categorical columns to reduce cardinality
  # Replace with your actual categorical column names
categorical_columns = ['type','nameOrig','nameDest']  
for col in categorical_columns:
    df[col] = reduce_cardinality(df[col])

# List of categorical columns
categorical_columns = ['type','nameOrig','nameDest']  # Replace with your actual categorical column names

# One-hot encode the categorical columns
df = pd.get_dummies(df, columns=categorical_columns)

# Preprocess the data
X = df.drop('isFraud', axis=1)  # Features (replace 'is_fraud' with your actual target column name)
y = df['isFraud']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save the model
joblib.dump(model, 'fraud_detection_model.pkl')
joblib.dump(X_train.columns.tolist(), 'model_features.pkl')
