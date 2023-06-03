import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# Load the dataset
data = pd.read_csv('aps_failure_training_set1.csv')

# Convert non-numeric columns to categorical data type
non_numeric_cols = data.select_dtypes(include=['object']).columns
data[non_numeric_cols] = data[non_numeric_cols].astype('category')

# Split the dataset into features and target variable
X = data.drop('class', axis=1)
y = data['class']

# One-hot encoding for categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_encoded = pd.DataFrame(encoder.fit_transform(data[non_numeric_cols]))

# Concatenate encoded features with numerical features
X_encoded.columns = encoder.get_feature_names(non_numeric_cols)
X = pd.concat([X, X_encoded], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the training data
imputer = SimpleImputer(strategy='constant', fill_value=-999)
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

# Train the XGBoost classifier
model = XGBClassifier()
model.fit(X_train, y_train)

# Create a prediction function
def predict_fault(X):
    X = pd.DataFrame(imputer.transform(X), columns=X.columns)
    predictions = model.predict(X)
    return predictions

# Streamlit app
def main():
    st.title("Truck APS Sensor Fault Detection")
    
    # User input
    st.sidebar.title("Input Data")
    st.sidebar.markdown("Enter the values of the features:")
    feature_names = X.columns
    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.sidebar.text_input(feature)
    
    # Predict button
    if st.sidebar.button("Predict"):
        input_data = pd.DataFrame([user_input])
        predictions = predict_fault(input_data)
        st.write("Predicted Result:", predictions[0])
    
if __name__ == "__main__":
    main()
