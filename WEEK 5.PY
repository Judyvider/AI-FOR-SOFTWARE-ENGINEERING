import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report

# --- 1. DATA SIMULATION (Mocking the Data Strategy) ---
def generate_hospital_data(n_samples=1000):
    np.random.seed(42)
    
    # Feature 1: Demographics
    ages = np.random.randint(18, 95, size=n_samples)
    zip_codes = np.random.choice(['Urban', 'Suburban', 'Rural'], size=n_samples, p=[0.5, 0.3, 0.2])
    
    # Feature 2: EHR Data (Vital Signs & Labs)
    # Introducing missing values to simulate real-world messiness
    blood_pressure = np.random.normal(120, 15, size=n_samples)
    blood_pressure[np.random.choice(n_samples, size=50)] = np.nan  # 50 missing values
    
    # Feature 3: Feature Engineering (Comorbidity Index)
    # 0 = Healthy, 5 = Multiple chronic conditions
    comorbidity_index = np.random.randint(0, 6, size=n_samples)
    
    # Feature 4: ADT Feeds (Length of Stay in days)
    length_of_stay = np.random.exponential(scale=5, size=n_samples).astype(int) + 1
    
    # Target Variable: Readmitted within 30 days (0 = No, 1 = Yes)
    # We make readmission correlated with Age and Comorbidity Index for realism
    prob = (ages * 0.005) + (comorbidity_index * 0.1) - 0.2
    prob = np.clip(prob, 0, 1)
    readmitted = np.random.binomial(1, prob)
    
    df = pd.DataFrame({
        'age': ages,
        'zip_type': zip_codes,
        'systolic_bp': blood_pressure,
        'comorbidity_index': comorbidity_index,
        'length_of_stay': length_of_stay,
        'readmitted': readmitted
    })
    
    return df

print("--- Step 1: Loading Data ---")
df = generate_hospital_data()
print(f"Dataset generated with {df.shape[0]} patients.")
print(df.head(), "\n")

# --- 2. PREPROCESSING PIPELINE (Addressing Data Strategy) ---

# Define features
numeric_features = ['age', 'systolic_bp', 'comorbidity_index', 'length_of_stay']
categorical_features = ['zip_type']

# Pipeline for Numeric Data:
# 1. Imputation: Fill missing Blood Pressure using KNN (as per assignment strategy)
# 2. Scaling: Standardize Age/BP to 0-1 range (Normalization)
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

# Pipeline for Categorical Data:
# 1. OneHotEncoding: Convert 'Urban/Rural' to numerical columns
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine into a single preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 3. MODEL DEVELOPMENT (Logistic Regression) ---

# We choose Logistic Regression for Interpretability (as per assignment justification)
# We add 'C=1.0' (L2 Regularization) to address Overfitting (Optimization Task)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, C=1.0, solver='liblinear'))
])

# Split Data
X = df.drop('readmitted', axis=1)
y = df['readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Step 2: Training Model ---")
model.fit(X_train, y_train)
print("Model trained successfully using Logistic Regression with L2 Regularization.\n")

# --- 4. EVALUATION ---

print("--- Step 3: Evaluation ---")
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"[TN: {cm[0][0]}, FP: {cm[0][1]}]")
print(f"[FN: {cm[1][0]}, TP: {cm[1][1]}]\n")

# Precision & Recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred))

# --- 5. DEPLOYMENT SIMULATION (API) ---

def risk_api_endpoint(patient_data):
    """
    Simulates the REST API described in the Deployment section.
    Takes raw JSON-like data, runs the pipeline, and returns a Risk Score.
    """
    # Convert dict to DataFrame for the pipeline
    input_df = pd.DataFrame([patient_data])
    
    # Get probability (Risk Score)
    # model.predict_proba returns [[prob_class_0, prob_class_1]]
    risk_probability = model.predict_proba(input_df)[0][1]
    risk_score = int(risk_probability * 100)
    
    alert = "SAFE"
    if risk_score > 80:
        alert = "HIGH RISK ALERT"
    elif risk_score > 50:
        alert = "MODERATE RISK"
        
    return {
        "patient_age": patient_data['age'],
        "risk_score": risk_score,
        "alert_level": alert
    }

print("--- Step 4: Simulated Deployment API Test ---")
new_patient = {
    'age': 82,
    'zip_type': 'Rural',
    'systolic_bp': 160, # High BP
    'comorbidity_index': 4, # High Comorbidity
    'length_of_stay': 2
}

api_response = risk_api_endpoint(new_patient)
print(f"Input: {new_patient}")
print(f"API Response: {api_response}")