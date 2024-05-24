import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑")

# Load the trained Random Forest model
with open(r'G:\DATA SCIENCE-25\SMALL_dataset\ML\supervised\classification\Diabetes-Prediction-App\model\SVM.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Define the Streamlit app
def main():
    # Sidebar menu
    with st.sidebar:
        selected = option_menu(
            'Prediction App',
            ['Description', 'Predict', 'About'],
            icons=['file-earmark-text', 'activity', 'info-circle'],
            menu_icon='hospital-fill',
            default_index=0
        )
        
        img = Image.open("G:\DATA SCIENCE-25\SMALL_dataset\ML\supervised\classification\Diabetes-Prediction-App\static\diabetes-01.jpg")
        st.image(img, width=290)
    
    st.header("🩺 Welcome to Diabetes Prediction App👉🏼")
    
    if selected == 'Description':
        st.subheader("Description")
        st.write("This application helps predict the likelihood of diabetes based health parameters using a machine learning model.")
        st.write()
        st.write("Diabetes is a chronic condition where blood glucose levels remain consistently high. It occurs either because the body doesn't produce enough insulin (Type 1) or can't use insulin effectively (Type 2).")
        st.write("High blood sugar can lead to serious health complications, including heart disease, kidney failure, and blindness.")
        st.write("Management involves medication, lifestyle changes, and regular monitoring to keep blood sugar levels in check.")
        st.write("Early diagnosis and proper management are crucial for preventing complications and maintaining overall health.")
        
    elif selected == 'Predict':
        st.text("Please input the following details and press the Predict button")

        # Getting the input data from the user
        col1, col2, col3 = st.columns(3)

        with col1:
            pregnancies = st.number_input("Enter the number of Pregnancies:", min_value=0, max_value=17)

        with col2:
            glucose = st.number_input("Enter the Glucose level:", min_value=0, max_value=200)

        with col3:
            blood_pressure = st.number_input("Enter the Blood Pressure:", min_value=0, max_value=150)

        with col1:
            skin_thickness = st.number_input("Enter the Skin Thickness:", min_value=0, max_value=100)

        with col2:
            insulin = st.number_input("Enter the Insulin Level:", min_value=0, max_value=1000)

        with col3:
            bmi = st.number_input("Enter the BMI:", min_value=0.0, max_value=70.0)

        with col1:
            diabetes_pedigree_function = st.number_input("Enter the Diabetes Pedigree Function:", min_value=0.0)

        with col2:
            age = st.number_input("Enter the Age:", min_value=0, max_value=120)
            
        # Create a feature DataFrame from user inputs
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree_function],
            'Age': [age]
        })

        # Predict using the model
        if st.button("Predict"):
            prediction = svm_model.predict(input_data)
            
            if prediction[0] == 0:
                st.success("The model predicts that the person is not diabetic.")
            else:
                st.success("The model predicts that the person is diabetic.")
            
    elif selected == 'About':
        st.subheader("About the App")
        st.write("This app uses a machine learning model to predict the likelihood of diabetes based on several health parameters. It is designed to assist in early diagnosis and management of diabetes.")

if __name__ == '__main__':
    main()
