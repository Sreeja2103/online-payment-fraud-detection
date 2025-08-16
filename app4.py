# Updating the app2.py script to load the correct model and provide additional features

improved_app2_code = """
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load(''fraud_detection_model.pkl'')

# Title of the app
st.title('Online Payment Fraud Detection')

# Collect user input
def user_input_features():
    transaction_type = st.selectbox('Transaction Type', ['CASH_IN', 'CASH_OUT', 'DEBIT', 'TRANSFER'])  # Adjust options as per your data
    amount = st.number_input('Amount')
    oldbalanceOrg = st.number_input('Old Balance Origin')
    newbalanceOrig = st.number_input('New Balance Origin')
    oldbalanceDest = st.number_input('Old Balance Destination')
    newbalanceDest = st.number_input('New Balance Destination')
    
    # Add more features as per your dataset
    data = {
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'type_' + transaction_type: 1
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Reindex input to match training features
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# Display user input
st.subheader('User Input:')
st.write(input_df)

# Prediction
prediction_proba = model.predict_proba(input_df)[0][1]

# Threshold slider
threshold = st.slider('Select a threshold for fraud detection', 0.0, 1.0, 0.5)

# Prediction based on the selected threshold
if prediction_proba > threshold:
    st.write('Prediction: **Fraudulent Transaction**')
else:
    st.write('Prediction: **Non-Fraudulent Transaction**')

# Show the probability of fraud
st.write(f'Probability of Fraud: {prediction_proba:.2f}')

# Display model performance metrics
st.subheader('Model Metrics:')
st.write(f'AUC-ROC: {model.oob_score_:.2f}')  # Example metric, assuming it was calculated earlier
"""

