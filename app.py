import streamlit as st
import joblib
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os

root = Path(__file__).parent

logger = logging.getLogger(__name__)

# Function to load the selected model
@st.cache_data # Use when function is expensive, has outside dependencies
def load_model(model_name):
    '''Loads Model from Artifacts Directory

    Args:
        model_name: Can either be rf for random forest or lr for logistic regression.
    '''
    try:
        file_name = str(model_name)+'_classifer.joblib'
        model_path = root / 'artifacts' / file_name
        logger.info('Model successfully loaded.')
        return joblib.load(model_path)
    except FileNotFoundError:
        logger.error('Input must be rf for random forest or lr for logistic regression.')
        return None

# Function to perform predictions using the loaded model
def predict(model, input_features):
    '''Performs predictions using loaded model

    Args:
        model: Model object used to make predictions.
        input_features: Feature vector the model will use to make prediction.
    '''
    try:
        prediction = model.predict(input_features)
        return prediction
    except TypeError:
        logger.error('Make sure that your inputs are numbers only.')
        return None

# Main function
def main():
    st.title("Cloud Classifier")
    
    st.write("""
    The Cloud Classifier is a web application built using streamlit. 
    If something catches your eye and you are wondering if it is a cloud or not, you are in the right place!
    Our machine learning models are highly performant and have been trained to tell whether or not an object is a cloud.
    Collect the following data about your object and enter it here, then watch as a prediction is made.  
    """)

    # Model selection dropdown
    model_version = st.selectbox("Select Model", ["rf", "lr"])

    # Load selected model
    model = load_model(model_version)

    # Feature inputs
    feature1 = st.number_input("Visible Contrast", value=0)
    feature2 = st.number_input("Visible Entropy", value=0)
    feature3 = st.number_input("IR Max", value=0)
    feature4 = st.number_input("IR Min", value=0)
    feature5 = st.number_input("IR Mean", value=0)

    # Perform prediction
    if st.button("Predict"):
        input_features = [[feature1, feature2, feature3, feature4, feature5]]
        prediction = predict(model, input_features)
        if prediction is [1]:
            st.write(f"Prediction: {prediction}, this is a cloud")
        else:
            st.write(f"Prediction: {prediction}, this is NOT a cloud")

if __name__ == "__main__":
    main()
