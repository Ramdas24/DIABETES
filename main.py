import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import joblib

# Load the trained LGBM model
model = joblib.load('lgbm_model.pkl')

# Load the label encoder for categorical columns
le = LabelEncoder()
# le = joblib.load('label_encoder.pkl')

# Streamlit app header
st.title("Diabetes Readmission Prediction")
import streamlit as st

# Define function for Home Page
def home():
    st.title("Welcome to Diabetes Prediction")
    st.write("Welcome to our Diabetes Prediction platform! We are dedicated to improving the effectiveness and accuracy of diabetes prediction models using advanced data science techniques. Explore our website to learn more about our research and how it can benefit healthcare practitioners.")
    st.write("To get started, click on the 'Diagnosis' page to make diabetes predictions based on patient data.")

# Define function for About Page
def about():
    st.title("About Us")
    st.header("Our Mission")
    st.write("Our mission is to enhance diabetes prediction efficiency through data-driven research. We analyze a decade of hospital care data for diabetic patients to improve diabetes management in healthcare settings.")
    st.header("Research Focus")
    st.write("Our research focuses on three key areas:")
    st.write("1. Data Imputation: We use advanced techniques to address missing data and restore completeness in patient records.")
    st.write("2. Feature Selection: We identify the most informative attributes for diabetes prediction, enhancing model efficiency and interpretability.")
    st.write("3. Ensemble Learning: We employ a combination of advanced machine learning algorithms to construct robust predictive models.")
    st.header("Impact")
    st.write("Our work aims to transform clinical data into actionable insights for healthcare practitioners, ultimately contributing to improved diabetes care in the field of health and medicine.")
    st.header("Contact Us")
    st.write("If you have any questions or would like to collaborate, please feel free to contact us.")

# Define function for Diagnosis Page
def diagnosis():
    st.title("Diabetes Diagnosis")
    st.write("Welcome to our Diabetes Diagnosis page. Here, you can input patient data and get predictions for diabetes. Please enter the required information below:")
    # User input section
    st.sidebar.header("User Input")

    # Input fields for each attribute based on your provided CSV columns
    encounter_id = st.sidebar.number_input("Encounter ID")
    patient_nbr = st.sidebar.number_input("Patient Number")
    race = st.sidebar.selectbox("None", ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "None"])
    age = st.sidebar.text_input("Age")
    weight = st.sidebar.text_input("Weight")
    gender = st.sidebar.text_input("Gender")
    admission_type_id = st.sidebar.selectbox("Admission Type ID", [1, 2, 3, 4, 5, 6, 7, 8])
    discharge_disposition_id = st.sidebar.selectbox("Discharge Disposition ID", [1, 2, 3, 4, 5, 6, 7, 8])
    admission_source_id = st.sidebar.selectbox("Admission Source ID", [1, 2, 3, 4, 5, 6, 7, 8])
    time_in_hospital = st.sidebar.number_input("Time in Hospital")
    num_lab_procedures = st.sidebar.number_input("Number of Lab Procedures")
    num_procedures = st.sidebar.number_input("Number of Procedures")
    num_medications = st.sidebar.number_input("Number of Medications")
    number_outpatient = st.sidebar.number_input("Number Outpatient")
    number_emergency = st.sidebar.number_input("Number Emergency")
    number_inpatient = st.sidebar.number_input("Number Inpatient")
    diag_1 = st.sidebar.text_input("Diagnosis 1")
    diag_2 = st.sidebar.text_input("Diagnosis 2")
    diag_3 = st.sidebar.text_input("Diagnosis 3")
    number_diagnoses = st.sidebar.number_input("Number of Diagnoses")
    max_glu_serum = st.sidebar.selectbox("Max Glu Serum", ["None", "Norm", ">200", ">300"])
    A1Cresult = st.sidebar.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])
    metformin = st.sidebar.selectbox("Metformin", ["No", "Steady", "Up", "Down"])
    repaglinide = st.sidebar.selectbox("Repaglinide", ["No", "Steady", "Up", "Down"])
    nateglinide = st.sidebar.selectbox("Nateglinide", ["No", "Steady", "Up", "Down"])
    chlorpropamide = st.sidebar.selectbox("Chlorpropamide", ["No", "Steady", "Up", "Down"])
    glimepiride = st.sidebar.selectbox("Glimepiride", ["No", "Steady", "Up", "Down"])
    acetohexamide = st.sidebar.selectbox("Acetohexamide", ["No", "Steady"])
    glipizide = st.sidebar.selectbox("Glipizide", ["No", "Steady", "Up", "Down"])
    glyburide = st.sidebar.selectbox("Glyburide", ["No", "Steady", "Up", "Down"])
    tolbutamide = st.sidebar.selectbox("Tolbutamide", ["No", "Steady"])
    pioglitazone = st.sidebar.selectbox("Pioglitazone", ["No", "Steady", "Up", "Down"])
    rosiglitazone = st.sidebar.selectbox("Rosiglitazone", ["No", "Steady", "Up", "Down"])
    acarbose = st.sidebar.selectbox("Acarbose", ["No", "Steady", "Up", "Down"])
    miglitol = st.sidebar.selectbox("Miglitol", ["No", "Steady", "Up", "Down"])
    troglitazone = st.sidebar.selectbox("Troglitazone", ["No", "Steady"])
    tolazamide = st.sidebar.selectbox("Tolazamide", ["No", "Steady"])
    examide = st.sidebar.selectbox("Examide", ["No"])
    citoglipton = st.sidebar.selectbox("Citoglipton", ["No"])
    insulin = st.sidebar.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
    glyburide_metformin = st.sidebar.selectbox("Glyburide Metformin", ["No", "Steady", "Up", "Down"])
    glipizide_metformin = st.sidebar.selectbox("Glipizide Metformin", ["No", "Steady"])
    glimepiride_pioglitazone = st.sidebar.selectbox("Glimepiride Pioglitazone", ["No", "Steady"])
    metformin_rosiglitazone = st.sidebar.selectbox("Metformin Rosiglitazone", ["No", "Steady"])
    metformin_pioglitazone = st.sidebar.selectbox("Metformin Pioglitazone", ["No", "Steady"])
    change = st.sidebar.selectbox("Change", ["No", "Ch"])
    diabetesMed = st.sidebar.selectbox("Diabetes Med", ["No", "Yes"])
    # readmitted = st.sidebar.selectbox("Readmitted", ["No", "<30", ">30"])

    # Predict button
    if st.sidebar.button("Predict"):
        # Prepare the input data as a DataFrame
        input_data = pd.DataFrame({
            'encounter_id': [encounter_id],
            'patient_nbr': [patient_nbr],
            'race':[race],
            'gender':[gender],
            'age': [age],
            'weight': [weight],
            'admission_type_id': [admission_type_id],
            'discharge_disposition_id': [discharge_disposition_id],
            'admission_source_id': [admission_source_id],
            'time_in_hospital': [time_in_hospital],
            'num_lab_procedures': [num_lab_procedures],
            'num_procedures': [num_procedures],
            'num_medications': [num_medications],
            'number_outpatient': [number_outpatient],
            'number_emergency': [number_emergency],
            'number_inpatient': [number_inpatient],
            'diag_1': [diag_1],
            'diag_2': [diag_2],
            'diag_3': [diag_3],
            'number_diagnoses': [number_diagnoses],
            'max_glu_serum': [max_glu_serum],
            'A1Cresult': [A1Cresult],
            'metformin': [metformin],
            'repaglinide': [repaglinide],
            'nateglinide': [nateglinide],
            'chlorpropamide': [chlorpropamide],
            'glimepiride': [glimepiride],
            'acetohexamide': [acetohexamide],
            'glipizide': [glipizide],
            'glyburide': [glyburide],
            'tolbutamide': [tolbutamide],
            'pioglitazone': [pioglitazone],
            'rosiglitazone': [rosiglitazone],
            'acarbose': [acarbose],
            'miglitol': [miglitol],
            'troglitazone': [troglitazone],
            'tolazamide': [tolazamide],
            'examide': [examide],
            'citoglipton': [citoglipton],
            'insulin': [insulin],
            'glyburide-metformin': [glyburide_metformin],
            'glipizide-metformin': [glipizide_metformin],
            'glimepiride-pioglitazone': [glimepiride_pioglitazone],
            'metformin-rosiglitazone': [metformin_rosiglitazone],
            'metformin-pioglitazone': [metformin_pioglitazone],
            'change': [change],
            'diabetesMed': [diabetesMed],
            
        })

        # Encode categorical columns
        for col in input_data.columns:
            # if col in le.classes_:
            input_data[col] = le.fit_transform(input_data[col])
        

        # Make predictions using the model
        prediction = model.predict(input_data)[0]

        # Map prediction to readmission status
        if prediction == 0:
            readmission_status = "Early readmission"
        elif prediction == 1:
            readmission_status = "Late readmission"
        else:
            readmission_status = "No readmission"

        # Display prediction result
        st.sidebar.subheader("Prediction Result")
        st.sidebar.write(f"The predicted readmission status is: {readmission_status}")
    st.write("Once you've entered the necessary data, click the 'Predict' button to receive the diagnosis.")

# Define function for Help Page
def help_page():
    st.title("Help and Support")
    st.header("How to Use")
    st.write("Our Diabetes Prediction platform is designed to be user-friendly. To make a diabetes prediction, follow these steps on the 'Diagnosis' page:")
    st.write("1. Enter the patient's encounter_id and patient_nbr.")
    st.write("2. Fill in the other required patient information, such as age, gender, and race.")
    st.write("3. Click the 'Predict' button to receive the diagnosis.")
    st.header("Troubleshooting")
    st.write("If you encounter any issues or have questions about using our platform, please check our Frequently Asked Questions (FAQs) section below:")
    # Add FAQs here
    st.write("### Frequently Asked Questions (FAQs)")
    st.write("1. **I'm getting an error when trying to predict. What should I do?**")
    st.write("- Please make sure all required fields are filled in correctly. Check for any typos or missing information.")
    st.write("2. **How accurate are the predictions?**")
    st.write("- Our predictions are based on advanced data science techniques and models. While we strive for accuracy, keep in mind that predictions may vary depending on the quality of the input data.")
    st.write("3. **Can I contact support for assistance?**")
    st.write("- Yes, you can reach out to our support team at [support@email.com](mailto:support@email.com) for assistance.")
    st.header("Contact Support")
    st.write("If you need further assistance or have specific questions, please don't hesitate to contact our support team.")
    # Add contact info for support

# Create a menu using radio buttons
menu = st.sidebar.radio("Menu", ["Home", "About", "Diagnosis", "Help"])

# Render the selected page
if menu == "Home":
    home()
elif menu == "About":
    about()
elif menu == "Diagnosis":
    diagnosis()
else:
    help_page()

