import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

# Create output folder if not exists
os.makedirs('datasets', exist_ok=True)

# Load data
df = pd.read_csv('heart.csv')

# Handle missing values
print("Missing Values:\n", df.isnull().sum())
df = df.dropna()

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Define columns
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Optional: log-transform skewed columns
X['chol'] = np.log1p(X['chol'])
X['oldpeak'] = np.log1p(X['oldpeak'])

# Define preprocessors
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Updated for scikit-learn ≥ 1.2
])

# Combine transformers
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Fit + transform
X_processed = preprocessor.fit_transform(X)

# Rebuild processed DataFrame
encoded_columns = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols)
processed_df = pd.DataFrame(X_processed, columns=numerical_cols + list(encoded_columns))
processed_df['target'] = y.values

# Save preprocessor for reuse in test time
joblib.dump(preprocessor, 'preprocessor.joblib')

# Stratified Train-Test Split
train_data, test_data = train_test_split(
    processed_df,
    test_size=0.2,
    stratify=processed_df['target'],
    random_state=42
)

# Shuffle training data before splitting into clients
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
split_data = np.array_split(train_data, 3)

# Save CSVs for clients and test data
for i, client_data in enumerate(split_data):
    client_data.to_csv(f'datasets/client_{i+1}_data.csv', index=False)
test_data.to_csv('datasets/test_data.csv', index=False)

print("✅ Data Preprocessing Completed and Saved with Improvements")
