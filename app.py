#pip install pyodbc
#pip install requests
#pip install fastapi
#pip install uvicorn
#pip install reportlab
#pip install letter
#pip install scikit-learn
#pip install scikit-surprise
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import numpy as np
import re
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import pyodbc
from io import BytesIO
from sklearn.decomposition import NMF
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from fastapi.responses import StreamingResponse
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
import tempfile

app = FastAPI()

# Allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Function to establish a connection to the Azure SQL Database
def connect_to_database():
    connection_string = f'DRIVER=ODBC Driver 18 for SQL Server;SERVER={"test2001.database.windows.net"};DATABASE={"Test"};UID={"Tariq"};PWD={"Khader@2001"}'
    return pyodbc.connect(connection_string)
connections = connect_to_database()
# Function to load datasets from the database
def load_datasets(connections):
    # Load patient dataset
    patients_df = pd.read_sql("SELECT * FROM dbo.Patients", connections)
    # Load medications dataset
    medications_df = pd.read_sql("SELECT * FROM dbo.Medications", connections)
    # Load drugs dataset
    drugs_df = pd.read_sql("SELECT * FROM dbo.DrugsLookups", connections)
    # Load blood pressure dataset
    blood_pressure_df = pd.read_sql("SELECT * FROM dbo.BloodPressureDATA", connections)
    # Load height dataset
    height_df = pd.read_sql("SELECT * FROM dbo.Heights", connections)
    # Load weight dataset
    weight_df = pd.read_sql("SELECT * FROM dbo.Weights", connections)
    # Load lab tests dataset
    lab_tests_df = pd.read_sql("SELECT * FROM dbo.Labs", connections)
    # Load comorbidities dataset
    comorbidities_df = pd.read_sql("SELECT * FROM dbo.Comorbidities", connections)
    # Load Last_Recommendation dataset
    MedicationsView_df = pd.read_sql("SELECT * FROM MedicationsView ORDER BY ISSUE_DATE;", connections)    
    # Load Labs_and_BP dataset
    Labs_and_BP_df = pd.read_sql("select PatientSourceID,FBS,HbA1C,RBS,Systolic,Diastolic from Labs_and_BP order by PatientSourceID,LabDate;", connections)
    # Load Users dataset
    Users_df = pd.read_sql("SELECT * FROM dbo.Users", connections)    
    
    return patients_df, medications_df, drugs_df, blood_pressure_df, height_df, weight_df, lab_tests_df, comorbidities_df, MedicationsView_df,Users_df

# Function to fetch patient data from the loaded datasets
def fetch_patient_data(patients_df, medications_df, drugs_df, blood_pressure_df, height_df, weight_df, lab_tests_df, comorbidities_df,MedicationsView_df, patient_id):
    # Fetch patient data from patients dataset
    patient_data = patients_df[patients_df["PatientSourceID"] == patient_id].iloc[0]

    # Fetch medications data for the patient
    medications_data = medications_df[medications_df["PatientSourceID"] == patient_id]

    # Fetch blood pressure data for the patient
    blood_pressure_data = blood_pressure_df[blood_pressure_df["PatientSourceID"] == patient_id]

    # Fetch height data for the patient
    height_data = height_df[height_df["PatientSourceID"] == patient_id]

    # Fetch weight data for the patient
    weight_data = weight_df[weight_df["PatientSourceID"] == patient_id]

    # Fetch lab tests data for the patient
    lab_tests_data = lab_tests_df[lab_tests_df["PatientSourceID"] == patient_id]

    # Fetch comorbidities data for the patient
    comorbidities_data = comorbidities_df[comorbidities_df["PatientSourceID"] == patient_id]
    
    # Perform the comparison
    drugs_data = drugs_df[drugs_df["DrugSourceID"].isin(medications_data["DRUG"])]

    MedicationsView_data = MedicationsView_df[MedicationsView_df["PatientSourceID"] == patient_id]
    
    return patient_data, medications_data, blood_pressure_data, height_data, weight_data, lab_tests_data, comorbidities_data, drugs_data, MedicationsView_data

# Function to generate recommendation
def generate_recommendation(patient_data, medications_data, blood_pressure_data, height_data, weight_data, lab_tests_data, comorbidities_data, drugs_data):
    recommendations = []

    # Extract patient information
    age = patient_data["Age"]
    sex = patient_data["Sex"]
    marital_status = patient_data["MaritalStatus"]
    diabetes = patient_data["DiabetesMellitus"]
    hypertension = patient_data["Hypertension"]

    # Extract comorbidities information
    comorbidities = comorbidities_data["Diagnosis"].tolist()
    
    # Generate recommendations based on patient demographics and health status
    if diabetes == 1:
        recommendations.append("-For diabetes management, focus on a balanced diet with plenty of fruits, vegetables, whole grains, and lean proteins. Avoid sugary drinks and snacks.")
        if age <= 12:
            recommendations.append("-Encourage outdoor activities like biking or playing sports to stay active and maintain healthy blood sugar levels.")
            recommendations.append("-Limit screen time and opt for physical games or activities to promote movement.")
            recommendations.append("-Hydrate with water instead of sugary beverages.")
        elif 13 <= age <= 18:
            recommendations.append("-Engage in regular exercise such as swimming, jogging, or playing sports to help manage blood sugar levels.")
            recommendations.append("-Choose nutritious snacks like fruits, nuts, or yogurt over processed snacks.")
            recommendations.append("-Monitor blood sugar levels regularly and keep a record of meals and physical activity.")
        elif 19 <= age <= 40:
            recommendations.append("-Incorporate strength training exercises like weightlifting or bodyweight exercises into your routine to improve insulin sensitivity and muscle health.")
            recommendations.append("-Experiment with different forms of physical activity such as dance classes, hiking, or martial arts to keep exercise enjoyable and engaging.")
            recommendations.append("-Focus on meal timing and distribution to help manage blood sugar levels, such as eating smaller, balanced meals every 3-4 hours.")
        elif 41 <= age <= 60:
            recommendations.append("-Explore mindful eating practices such as paying attention to hunger cues, practicing portion control, and savoring each bite to promote better blood sugar regulation.")
            recommendations.append("-Incorporate yoga or tai chi into your routine to reduce stress levels and improve overall well-being, which can positively impact diabetes management.")
            recommendations.append("-Consider consulting with a registered dietitian to create a personalized meal plan that aligns with your diabetes management goals and lifestyle preferences.")
        else:
            recommendations.append("-Prioritize regular physical activity that includes a combination of aerobic exercises, strength training, and flexibility exercises to support overall health and blood sugar control.")
            recommendations.append("-Stay informed about advances in diabetes management technologies such as continuous glucose monitoring systems and automated insulin delivery systems to enhance treatment outcomes.")
            recommendations.append("-Join a diabetes support group or online community to connect with others living with diabetes, share experiences, and gain valuable support and resources.")

    if hypertension == 1:
        recommendations.append("-To manage hypertension, focus on a low-sodium diet rich in fruits, vegetables, whole grains, and lean proteins.")
        if age <= 12:
            recommendations.append("-Encourage outdoor play and physical activities to promote heart health and manage blood pressure.")
            recommendations.append("-Limit salty snacks and processed foods, opting for healthier options like fruits and vegetables.")
            recommendations.append("-Stay hydrated by drinking water throughout the day.")
        elif 13 <= age <= 18:
            recommendations.append("-Participate in regular physical activities such as biking, dancing, or jogging to support heart health and control blood pressure.")
            recommendations.append("-Avoid excess caffeine and sugary beverages, which can elevate blood pressure.")
            recommendations.append("-Practice stress-reducing techniques like deep breathing or meditation to support heart health.")
        elif 19 <= age <= 40:
            recommendations.append("-Experiment with different cooking methods such as grilling, steaming, or baking to prepare flavorful meals without adding excess salt.")
            recommendations.append("-Incorporate interval training or circuit workouts into your exercise routine to improve cardiovascular fitness and lower blood pressure.")
            recommendations.append("-Try incorporating mindfulness practices such as guided imagery or progressive muscle relaxation into your daily routine to manage stress and promote relaxation.")
        elif 41 <= age <= 60:
            recommendations.append("-Include potassium-rich foods such as bananas, spinach, and avocados in your diet, as potassium can help counteract the effects of sodium on blood pressure.")
            recommendations.append("-Engage in activities that promote relaxation and stress reduction, such as gardening, listening to music, or spending time in nature.")
            recommendations.append("-Monitor your blood pressure regularly at home and keep a log to track changes over time, which can help inform treatment decisions and lifestyle modifications.")
        else:
            recommendations.append("-Incorporate heart-healthy fats from sources such as nuts, seeds, and olive oil into your diet to help support cardiovascular health and lower blood pressure.")
            recommendations.append("-Practice mindful eating by slowing down during meals, chewing food thoroughly, and paying attention to hunger and fullness cues to prevent overeating.")
            recommendations.append("-Explore alternative therapies such as acupuncture, biofeedback, or massage therapy to complement traditional treatments for hypertension and promote overall well-being.")

    # Additional conditions based on datasets
    if not medications_data.empty:
        recommendations.append("-Remember to take your medications as prescribed to support your overall health.")
        
     # Additional recommendations based on medications
    # Check if the drug is in medications_data
    if (medications_data["DRUG"].astype(int).isin(drugs_data["DrugSourceID"].astype(int))).any():
        # Check if the drug is also in drugs_data
        if (drugs_data["MainSideEffect"] == "Immunosuppressants").any() and (medications_data["QTY"] > 0).any():
            recommendations.append("-Ensure adherence to your medication regimen and report any adverse effects to your healthcare provider for further evaluation.")

    if not blood_pressure_data.empty:
        # Get the newest blood pressure readings
        blood_pressure_data = blood_pressure_data.sort_values(by="DateTimeTaken", ascending=False).iloc[0]
        systolic = blood_pressure_data["Systolic"]
        diastolic = blood_pressure_data["Diastolic"]
        
        if 130 <= systolic <= 139 or 80 <= diastolic <= 89:
            recommendations.append("-Your blood pressure is in Stage 1 hypertension range. Focus on lifestyle modifications such as diet and exercise to manage it.")
        elif systolic >= 140 or diastolic >= 90:
            recommendations.append("-Your blood pressure is in Stage 2 hypertension range. It's important to consult with your healthcare provider for proper management and treatment.")
        else:
            recommendations.append("-Your blood pressure is within healthy ranges. Continue monitoring it regularly.")

    if not weight_data.empty and not height_data.empty:
        # Get the newest weight and height readings
        weight_data = weight_data.sort_values(by="DateTimeTaken", ascending=False).iloc[0]
        height_data = height_data.sort_values(by="DateTimeTaken", ascending=False).iloc[0]
        
        bmi = (float(weight_data["RATE"]) / (float(height_data["RATE"]) ** 2)) * 703 # Calculate BMI
        if bmi < 18.5:
            recommendations.append("-Your BMI indicates underweight. It's important to focus on gaining weight through a balanced diet rich in nutrient-dense foods and engaging in strength training exercises to build muscle mass.")
            recommendations.append("-Consider consulting with a dietitian to create a meal plan tailored to your needs and goals.")
        elif 18.5 <= bmi <= 25:
            recommendations.append("-Your BMI is within the healthy range. Continue to maintain a balanced diet and regular exercise routine to support overall health.")
        elif bmi > 25:
            recommendations.append("-Your BMI indicates overweight or obesity. It's essential to focus on weight management through dietary modifications and regular exercise.")
            recommendations.append("-Recommend incorporating a variety of physical activities such as cardio, strength training, and flexibility exercises into your routine.")
            recommendations.append("-Encourage adopting healthy eating habits such as portion control, reducing intake of processed foods and sugary beverages, and increasing consumption of fruits, vegetables, and lean proteins.")

    if not lab_tests_data.empty:
        # Get the newest LabTestResult
        lab_tests_data = lab_tests_data.sort_values(by="LabTestDatetimeTaken", ascending=False).iloc[0]
        if lab_tests_data["LabTestType"] == "HbA1C" and lab_tests_data["LabTestResult"] > 0:
            if lab_tests_data["LabTestResult"] < 7:
                recommendations.append("-Maintain your current diabetes management plan, including dietary modifications and regular physical activity, to keep HbA1c levels stable.")
            else:
                recommendations.append("-Consider reviewing your diabetes management strategies to optimize blood sugar control and improve HbA1c levels.")
        elif lab_tests_data["LabTestType"] == "FBS" and lab_tests_data["LabTestResult"] > 0:
            if lab_tests_data["LabTestResult"] > 126:
                recommendations.append("-If fasting blood sugar levels are elevated, focus on dietary changes and exercise to help manage blood sugar levels.")
                recommendations.append("-Consider scheduling an appointment to discuss these results and potential treatment options.")
            else:
                recommendations.append("-Discuss your lab test results to identify areas for improvement and develop a plan for optimizing your health.")
        elif lab_tests_data["LabTestType"] == "RBS" and lab_tests_data["LabTestResult"] > 0:
            if lab_tests_data["LabTestResult"] > 200:
                recommendations.append("-Your random blood sugar levels are elevated. It's important to monitor your diet and lifestyle habits.")
                recommendations.append("-Consider scheduling an appointment with a healthcare provider for further evaluation and guidance.")
            else:
                recommendations.append("-Continue monitoring your random blood sugar levels and follow a balanced diet to maintain stable blood sugar levels.")

    # Include recommendations based on comorbidities
    for condition in comorbidities:
        if "HYPERTENSION" in condition.upper():
            recommendations.append("-You have a comorbidity of hypertension. It's important to manage your blood pressure through lifestyle modifications such as diet, exercise, and medication adherence.")
            recommendations.append("-Follow a diet rich in fruits, vegetables, whole grains, and lean proteins, and limit sodium intake.")
            recommendations.append("-Engage in regular physical activity, such as brisk walking, swimming, or cycling, to support heart health.")
            recommendations.append("-Monitor your blood pressure regularly and consult with your healthcare provider for personalized guidance.")
        elif "DIABETES" in condition.upper():
            recommendations.append("-You have a comorbidity of diabetes. Focus on maintaining stable blood sugar levels through diet, exercise, regular monitoring, and medication adherence.")
            recommendations.append("-Choose carbohydrates that have a low glycemic index to prevent spikes in blood sugar levels.")
            recommendations.append("-Incorporate regular exercise into your routine, such as aerobic activities, strength training, and flexibility exercises.")
            recommendations.append("-Monitor your blood sugar levels closely and consult with your healthcare provider for adjustments to your diabetes management plan.")
        elif "CVA" in condition.upper():
            recommendations.append("-You have a history of cerebrovascular accident (CVA). It's important to prioritize heart health and reduce risk factors for future cardiovascular events.")
            recommendations.append("-Follow a heart-healthy diet that includes plenty of fruits, vegetables, whole grains, and lean proteins.")
            recommendations.append("-Engage in regular physical activity to improve circulation and overall cardiovascular health.")
            recommendations.append("-Take medications as prescribed by your healthcare provider to manage risk factors such as high blood pressure and high cholesterol.")
        elif "B-COMPLEX DEFIC NEC" in condition.upper():
            recommendations.append("-You have a deficiency in B-complex vitamins. Focus on incorporating foods rich in B vitamins into your diet to address this deficiency.")
            recommendations.append("-Include sources of B vitamins such as whole grains, leafy greens, legumes, nuts, seeds, and lean meats in your meals.")
            recommendations.append("-Consider taking a B-complex supplement under the guidance of a healthcare professional to ensure adequate intake.")
        elif "CHR APICAL PERIODONTITIS" in condition.upper():
            recommendations.append("-You have chronic apical periodontitis. Good oral hygiene is crucial to prevent further inflammation and infection.")
            recommendations.append("-Brush your teeth at least twice a day and floss daily to remove plaque and bacteria from the teeth and gums.")
            recommendations.append("-Schedule regular dental check-ups and cleanings to monitor the condition of your teeth and gums.")
        elif "FOLLOW-UP EXAM NEC" in condition.upper():
            recommendations.append("-You have a follow-up exam scheduled. Make sure to attend the appointment and discuss any concerns or questions with your healthcare provider.")
            recommendations.append("-Prepare for the appointment by keeping track of your symptoms, medications, and any changes in your health since your last visit.")
            recommendations.append("-Ask about any recommended screenings or tests to monitor your condition and stay proactive about your health.")
        elif "ACUTE UPPER RESPIRATORY INFECTION, UNSPECIFIED" in condition.upper():
            recommendations.append("-You have been diagnosed with acute upper respiratory infection. Rest, stay hydrated, and consider over-the-counter medications for symptom relief.")
            recommendations.append("-Practice good hand hygiene and respiratory etiquette to prevent the spread of infection to others.")
        elif "TYPE 2 DIABETES MELLITUS WITH HYPERGLYCEMIA" in condition.upper():
            recommendations.append("-You have been diagnosed with type 2 diabetes mellitus with hyperglycemia. Follow your healthcare provider's instructions for managing your blood sugar levels.")
            recommendations.append("-Monitor your blood sugar regularly and take medications as prescribed.")
            recommendations.append("-Adopt a healthy lifestyle with a balanced diet, regular exercise, and stress management techniques.")
        elif "ACUTE PAIN DUE TO TRAUMA" in condition.upper():
            recommendations.append("-You are experiencing acute pain due to trauma. Follow your healthcare provider's recommendations for pain management, including medication, rest, and ice or heat therapy.")
            recommendations.append("-Seek medical attention if the pain persists or worsens, or if you experience any concerning symptoms.")
        elif "LOW BACK PAIN" in condition.upper():
            recommendations.append("-You have been diagnosed with low back pain. Practice good posture, use proper lifting techniques, and engage in gentle stretching and strengthening exercises to alleviate discomfort.")
            recommendations.append("-Consider physical therapy or chiropractic care for targeted treatment and pain relief.")
        elif "COUGH" in condition.upper():
            recommendations.append("-You are experiencing a cough. Stay hydrated, get plenty of rest, and consider over-the-counter remedies such as cough drops or cough syrup for relief.")
            recommendations.append("-If the cough persists for more than a few days or is accompanied by other symptoms, consult with your healthcare provider.")
        elif "OTHER WAITING PERIOD FOR INVESTIGATION AND TREATMENT" in condition.upper():
            recommendations.append("-You are waiting for investigation and treatment. Stay informed about your upcoming appointments or procedures, and follow any pre-appointment instructions provided by your healthcare provider.")
            recommendations.append("-If you have any concerns or questions about the waiting period, don't hesitate to reach out to your healthcare provider for clarification.")
        elif "ENCOUNTER FOR OTHER GENERAL EXAMINATION" in condition.upper():
            recommendations.append("-You have an encounter scheduled for a general examination. Make sure to attend the appointment and discuss any health concerns or questions with your healthcare provider.")
            recommendations.append("-Prepare for the appointment by compiling a list of medications, symptoms, and any changes in your health since your last visit.")
        elif "ILLNESS, UNSPECIFIED" in condition.upper():
            recommendations.append("-You have been diagnosed with an unspecified illness. Follow your healthcare provider's recommendations for symptom management and treatment.")
            recommendations.append("-If you experience any concerning symptoms or if your condition worsens, seek medical attention promptly.")
        elif "URINARY TRACT INFECTION, SITE NOT SPECIFIED" in condition.upper():
            recommendations.append("-You have been diagnosed with a urinary tract infection. Drink plenty of water, avoid irritants like caffeine and alcohol, and take prescribed antibiotics as directed.")
            recommendations.append("-Practice good hygiene, including wiping from front to back, to prevent the spread of bacteria.")
        elif "SHORTNESS OF BREATH" in condition.upper():
            recommendations.append("-You are experiencing shortness of breath. Rest, avoid triggers like smoke or allergens, and seek medical attention if the symptoms are severe or persistent.")
            recommendations.append("-If you have a known respiratory condition, follow your healthcare provider's instructions for managing symptoms and seek prompt medical care if needed.")

    return recommendations


# Load datasets
patients_df, medications_df, drugs_df, blood_pressure_df, height_df, weight_df, lab_tests_df, comorbidities_df,MedicationsView_df,Users_df = load_datasets(connections)

patient_id = None
recommendation = None
patient_data = None
medications_data = None
blood_pressure_data = None
height_data = None
weight_data = None
lab_tests_data = None
comorbidities_data = None
drugs_data = None
MedicationsView_data = None
rec_med = None
rec_dos = None
meds_list = None

# Define a function to get the first element of a DataFrame column safely
def get_first_element(df, column_name,date_column):
    try:
        sorted_df = df.sort_values(by=date_column)
        return sorted_df[column_name].iloc[-1]
    except IndexError:
        return "Unknown"

# Define a function to get blood pressure data safely
def get_blood_pressure(bp_data):
    try:
        sorted_bp_data = bp_data.sort_values(by='DateTimeTaken')
        systolic = sorted_bp_data['Systolic'].iloc[-1]
        diastolic = sorted_bp_data['Diastolic'].iloc[-1]
        return f"{systolic}/{diastolic}"
    except IndexError:
        return "Unknown"
def convert_numpy_types(data):
    """
    Convert numpy data types to native Python types.
    """
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    else:
        return data

logging.basicConfig(level=logging.INFO)
class User(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(user: User):
    logging.info("Received login request: %s", user)
    # Check if the provided username exists in the dataset
    if user.username in Users_df['Username'].values:
        # Find the row corresponding to the provided username
        user_row = Users_df[Users_df['Username'] == user.username].iloc[0]
        # Check if the provided password matches the stored password for the username
        if user_row['Password'] == user.password:
            return {"message": "Login successful"}
        else:
            # If passwords don't match, raise an HTTPException with status code 401 (Unauthorized)
            raise HTTPException(status_code=401, detail="Invalid username or password")
    else:
        # If username not found, raise an HTTPException with status code 401 (Unauthorized)
        raise HTTPException(status_code=401, detail="Username not found")

#Data Analysis 

# Preprocessing Weghits and Heigts
weight_df = weight_df.loc[weight_df['RATE'] != 'Unavailable'].copy()
weight_df['RATE'] = weight_df['RATE'].astype('float')

weight_df = weight_df.loc[weight_df['RATE'] > 25]

#Preprocessing Drugs

drugs_df = drugs_df[~drugs_df['DrugGenericName'].isin(['N', 'ZZ'])]
drugs_df['DrugGenericName'] = drugs_df['DrugGenericName'].str.replace(r'z+\s*', ' ', regex=True)

patterns = ['ZZ ZZZ', 'ZZZ', 'ZZZZZ', 'ZZZZZZZZZZZZZ']

drugs_df = drugs_df[~drugs_df['DrugGenericName'].str.contains('|'.join(patterns))]

valid_drugs = set(drugs_df['DrugSourceID'])

meds_R_filtered = medications_df[medications_df['DRUG'].isin(valid_drugs)]
medications_df = meds_R_filtered
#Preprocessing patients

patients_df['MaritalStatus'].replace({'Foreign wife-jor son': 'Married', 'Marriage from foreign': 'Married',
                                              'Miss wife': 'Married', 'Widoed': 'Widowed', 'Widowed foreign': 'Widowed',
                                              'Wife of an absent': 'Married'}, inplace = True)
patients_df = patients_df[patients_df['MaritalStatus'] != 'Others']
patients_df = patients_df[patients_df['MaritalStatus'] != 'Dead']

mapping = {
    'Single': 0b00,  # Binary 00 is 0 in decimal
    'Married': 0b01,  # Binary 01 is 1 in decimal
    'Divorced': 0b10,  # Binary 10 is 2 in decimal
    'Widowed': 0b11  # Binary 11 is 3 in decimal
}

patients_df['MaritalStatus'] = patients_df['MaritalStatus'].map(mapping)

patients_df['Sex'].replace({'Male': 0, 'Female': 1}, inplace = True)

#Preprocessing Labs
def filter_lab_tests(row):
    if row['LabTestType'] == 'HbA1C':
        return 3.4 <= row['LabTestResult'] <= 20.2
    elif row['LabTestType'] == 'FBS':
        return 40 <= row['LabTestResult'] <= 1000
    elif row['LabTestType'] == 'RBS':
        return 70 <= row['LabTestResult'] <= 1000
    return False

lab_tests_df = lab_tests_df[lab_tests_df.apply(filter_lab_tests, axis = 1)]

blood_pressure_df = blood_pressure_df[blood_pressure_df['Diastolic'] >= 55]

#Preprocessing Comorbidities

comorbidities_df.loc[comorbidities_df['Diagnosis'].str.contains('DMII'), 'Diagnosis'] = comorbidities_df['Diagnosis'].str.replace('DMII', 'Type 2 diabetes mellitus')

#Datasets for Generate Medications and Dosages Names

drugs_lookup_R = drugs_df
patients_R = patients_df 
weights_R = weight_df
heights_R = height_df
meds_R = medications_df
labs_R = lab_tests_df
readings_R = blood_pressure_df
diseases_R = comorbidities_df
#Generate Medications and Dosages Names
def is_cream(drug_name):
    cream_keywords = ['oint', 'cream', 'top', 'opht', 'soln', 'ophth', 'gel']
    drug_name_lower = drug_name.lower()
    return any(re.search(r'\b' + keyword + r'\b', drug_name_lower) for keyword in cream_keywords)

def extract_dosage(drug_name):
    if is_cream(drug_name):
        return 'Topical Nature'

    drug_name_lower = drug_name.lower()

    if re.search(r'\bpen\b', drug_name_lower) or re.search(r'\bpenfill\b', drug_name_lower):
        match = re.search(r'\d+.*$', drug_name)
        if match:
            return match.group(0)

    if '/' in drug_name or '%' in drug_name:
        match = re.search(r'\d+\.?\d*[A-Za-z]*\/\d*\.?\d*[A-Za-z]*\s[A-Za-z]+\s\d+\.?\d*[A-Za-z]*', drug_name)
        if match:
            return match.group(0)

    match = re.search(r'\d+.*$', drug_name)
    if match:
        return match.group(0)

    parts = drug_name.split('/')
    dosages = []
    for part in parts:
        match = re.search(r'(?<=\s)\d+.*$', part)
        if match:
            dosages.append(match.group(0))

    if dosages:
        return '/'.join(dosages)

    return None

def extract_medication(drug_name):
    if is_cream(drug_name):
        return drug_name

    drug_name_lower = drug_name.lower()

    if re.search(r'\bpen\b', drug_name_lower) or re.search(r'\bpenfill\b', drug_name_lower):
        return re.sub(r'\d+.*$', '', drug_name).strip()

    if '/' in drug_name or '%' in drug_name:
        match = re.search(r'\d+\.?\d*[A-Za-z]*\/\d*\.?\d*[A-Za-z]*\s[A-Za-z]+\s\d+\.?\d*[A-Za-z]*', drug_name)
        if match:
            return re.sub(r'\d+\.?\d*[A-Za-z]*\/\d*\.?\d*[A-Za-z]*\s[A-Za-z]+\s\d+\.?\d*[A-Za-z]*', '', drug_name).strip()

    name = re.sub(r'\s*\d+.*$', '', drug_name).strip()

    parts = drug_name.split('/')
    names = []
    for part in parts:
        clean_part = re.sub(r'\s*\d+.*$', '', part).strip()
        if clean_part:
            names.append(clean_part)

    return '/'.join(names)

def fix_name(drug_name):
    drug_name = re.sub(' ,', ', ', str(drug_name))
    drug_name = re.sub(r'/*$', '', str(drug_name))
    drug_name = re.sub(r' $', '', str(drug_name))
    return drug_name

drugs_lookup_R['dosage'] = drugs_lookup_R['DrugGenericName'].apply(extract_dosage)
drugs_lookup_R['drug_name'] = drugs_lookup_R['DrugGenericName'].apply(extract_medication)
drugs_lookup_R['drug_name'] = drugs_lookup_R['drug_name'].apply(fix_name)

def get_med_dos(drugID, df):
    med = df.loc[df['DrugSourceID'] == drugID, 'drug_name'].values[0]
    dos = df.loc[df['DrugSourceID'] == drugID, 'dosage'].values[0]
    return med, dos
#################################################################################################

# Datasets for Matrices 
patients = patients_df 
com = comorbidities_df
meds = medications_df
labs = lab_tests_df
new_labs = lab_tests_df
drugs_lookup = drugs_df
weights = weight_df
heights = height_df
bp = blood_pressure_df
# preprocessing for matrices
patients = pd.merge(patients, weights.drop(['DomainID', 'DateTimeTaken'], axis=1), on='PatientSourceID', how='left')
patients.rename(columns={'RATE': 'weight'},inplace = True)
patients = pd.merge(patients,heights.drop(['DomainID', 'DateTimeTaken'], axis=1), on='PatientSourceID', how='left')
patients.rename(columns={'RATE': 'heights'}, inplace=True)

# Identify patients with both weight and height or without weight and height
patients_with_both = patients.dropna(subset=['weight', 'heights'])
patients_with_either = patients[(patients['weight'].isna() | patients['heights'].isna())]

# Identify patients with weight and height but without medications
patients_with_medications = meds['PatientSourceID'].unique()
patients_with_weight_height_no_meds = patients_with_both[~patients_with_both['PatientSourceID'].isin(patients_with_medications)]['PatientSourceID']

# Identify patients with weight and height but without labs or bp readings
patients_with_data = set(new_labs['PatientSourceID']).union(set(bp['PatientSourceID']))
patients_with_weight_height_no_data = patients_with_both[~patients_with_both['PatientSourceID'].isin(patients_with_data)]['PatientSourceID']

# Combine patients to drop
patients_to_drop = patients_with_either['PatientSourceID'].tolist() + patients_with_weight_height_no_meds.tolist() + patients_with_weight_height_no_data.tolist()

# Drop patients from all tables
patients = patients[~patients['PatientSourceID'].isin(patients_to_drop)]
meds = meds[meds['PatientSourceID'].isin(patients_with_medications)]  # Filter medications to keep only patients with medications
new_labs = new_labs[~new_labs['PatientSourceID'].isin(patients_to_drop)]
bp = bp[~bp['PatientSourceID'].isin(patients_to_drop)]

# Get unique PatientSourceID values from the patients table
valid_patients = patients['PatientSourceID'].unique()

# Filter the com table to keep only patients present in the patients table
meds = meds[meds['PatientSourceID'].isin(valid_patients)]
# Copy of patient_profile_matrix
patient_profile_matrix = patients
#Create BP matrix
# Convert DatetimeTaken to datetime
bp['DateTimeTaken'] = pd.to_datetime(bp['DateTimeTaken'])

# Sort by PatientSourceID and DateTimeTaken
df = bp.sort_values(by=['PatientSourceID', 'DateTimeTaken'])

# Drop duplicates to keep the latest reading for each patient
latest_bp_readings = df.drop_duplicates(subset=['PatientSourceID'], keep='last')

# Select relevant columns
patient_bp_matrix = latest_bp_readings[['PatientSourceID', 'BPReading', 'Systolic', 'Diastolic']]

# Get unique PatientSourceID values from the patients table
valid_patients = patients['PatientSourceID'].unique()

# Filter the com table to keep only patients  present in the patients table
com = com[com['PatientSourceID'].isin(valid_patients)]

#creating patient diagnosis matrix

diagnosis_matrix = pd.get_dummies(com['Diagnosis'])

# Concatenate PatientSourceID with diagnosis_matrix
patient_diagnosis_matrix = pd.concat([com['PatientSourceID'], diagnosis_matrix], axis=1)

# Group by PatientSourceID and sum to handle multiple diagnoses per patient
patient_diagnosis_matrix = patient_diagnosis_matrix.groupby('PatientSourceID').sum()

# Replace non-zero values with 1 to indicate presence of diagnosis
patient_diagnosis_matrix = patient_diagnosis_matrix.applymap(lambda x: 1 if x > 0 else 0)

#create Patient_medication matrix
patient_medication_matrix = pd.crosstab(meds['PatientSourceID'], meds['DRUG']).clip(upper=1)

# Reset the index to turn the index 'PatientSourceID' into a regular column
patient_medication_matrix.reset_index(inplace=True)

# Ensure the columns are handled correctly if there's an issue with multi-level column handling
if isinstance(patient_medication_matrix.columns, pd.MultiIndex):
    patient_medication_matrix.columns = ['_'.join(col).strip() if col[0] else col[1] for col in patient_medication_matrix.columns.values]
else:
    patient_medication_matrix.columns = [col if isinstance(col, str) else str(col) for col in patient_medication_matrix.columns]

#creating patient-labs matrix

new_labs['LabTestDatetimeTaken'] = pd.to_datetime(new_labs['LabTestDatetimeTaken']).dt.date

df = new_labs
df = df.sort_values(by=['PatientSourceID', 'LabTestDatetimeTaken'])

# Calculate the difference of values for each lab test type
df['ResultDifference'] = df.groupby(['PatientSourceID', 'LabTestType'])['LabTestResult'].diff()

# Filter to get the last two entries for each lab test type per patient
df_filtered = df.groupby(['PatientSourceID', 'LabTestType']).tail(2)

# Pivot the DataFrame to create the patient-labs matrix
patient_labs_matrix = df_filtered.pivot_table(
    index='PatientSourceID',
    columns='LabTestType',
    values=['LabTestResult', 'ResultDifference'],
    aggfunc='last'
)

# Reset the index to convert 'PatientSourceID' from an index to a column
patient_labs_matrix.reset_index(inplace=True)

# Flatten the column names to make them more manageable
patient_labs_matrix.columns = [f'{col[1]}_{col[0]}' if col[1] else col[0] for col in patient_labs_matrix.columns]

patient_labs_matrix.fillna(0, inplace=True)

# Merge all matrices on 'PatientSourceID'
merged_data = pd.merge(patient_profile_matrix, patient_medication_matrix, on='PatientSourceID')
merged_data = pd.merge(merged_data, patient_diagnosis_matrix, on='PatientSourceID')
merged_data = pd.merge(merged_data, patient_labs_matrix, on='PatientSourceID')
merged_data = pd.merge(merged_data, patient_bp_matrix, on='PatientSourceID')

def convert_to_surprise_format(df):
    # Convert DataFrame to long format
    long_df = df.melt(id_vars='PatientSourceID', var_name='ItemID', value_name='Taken')

    # Ensure 'Taken' column contains numerical values
    long_df['Taken'] = pd.to_numeric(long_df['Taken'], errors='coerce')

    # Remove rows where 'Taken' column contains NaN or non-numeric values
    long_df = long_df.dropna(subset=['Taken'])

    return long_df[['PatientSourceID', 'ItemID', 'Taken']]
def test(merged_data):
# Convert merged DataFrame to Surprise format
    surprise_data = convert_to_surprise_format(merged_data)
    # Define the scale of the dataset
    reader = Reader(rating_scale=(0, 1))

    # Load the dataset into Surprise
    data = Dataset.load_from_df(surprise_data[['PatientSourceID', 'ItemID', 'Taken']], reader)

    # Split data into training and test sets
    trainset, testset = train_test_split(data, test_size=0.25)

    # Use the SVD algorithm
    algo = SVD()

    # Train the algorithm on the trainset and predict ratings for the testset
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Calculate and print the RMSE
    rmse = accuracy.rmse(predictions)

    return algo, trainset

def get_item_recommendations(patient_id, algo, trainset, drugs_lookup, patient_profile_matrix):
    meds_list = []
    # Get patient's medical conditions
    patient_row = patient_profile_matrix[patient_profile_matrix['PatientSourceID'] == patient_id]
    if patient_row.empty:
        print("Patient not found in the profile matrix.")
        return []

    has_diabetes = patient_row.iloc[0]['DiabetesMellitus'] == 1
    has_hypertension = patient_row.iloc[0]['Hypertension'] == 1

    # Get a list of all items
    item_ids = trainset.all_items()
    item_labels = [trainset.to_raw_iid(x) for x in item_ids]

    # Predict scores for all items
    predictions = [algo.predict(patient_id, iid).est for iid in item_labels]

    # Combine IDs and scores
    item_scores = list(zip(item_labels, predictions))

    # Filter out non-numeric item IDs (assuming drug IDs are numeric)
    numeric_item_scores = [(item, score) for item, score in item_scores if item.isdigit()]

    # Sort the items based on the predicted scores
    numeric_item_scores.sort(key=lambda x: x[1], reverse=True)

    # Print top recommended items, filtering based on patient's conditions
    for item, score in numeric_item_scores[:100]:
        row = drugs_lookup[drugs_lookup['DrugSourceID'] == int(item)]
        if not row.empty:
            usage = row.iloc[0]['Usage']
            if (has_diabetes and usage == "Diabetes Mellitus") or (has_hypertension and usage == "Arrhythmia and Hypertension"):
                meds_list.append((item,usage))
    hypertension_drug = None
    diabetes_drug = None
    for item,usage in meds_list:
        if usage == 'Diabetes Mellitus':
                diabetes_drug = item
                break

    for item,usage in meds_list:
        if usage == 'Arrhythmia and Hypertension':
                hypertension_drug = item
                break

    rec_med=[]
    rec_dos=[]

    if hypertension_drug:
            medication_h, dosage_h = get_med_dos(int(hypertension_drug), drugs_lookup_R)
            rec_med.append(str(medication_h))
            rec_dos.append(dosage_h)
    if diabetes_drug:
            medication_d, dosage_d = get_med_dos(int(diabetes_drug), drugs_lookup_R)
            rec_med.append(str(medication_d))
            rec_dos.append(dosage_d)
    return rec_med,rec_dos,meds_list



def reset_global_variables():
    global patient_id, recommendation, patient_data, medications_data, blood_pressure_data, height_data, weight_data, lab_tests_data, comorbidities_data, drugs_data, MedicationsView_data,recommendation_medication,recommendation_dosage,rec_med,rec_dos,meds_list,last_med,last_dos
    
    patient_id = None
    recommendation = None
    patient_data = None
    medications_data = None
    blood_pressure_data = None
    height_data = None
    weight_data = None
    lab_tests_data = None
    comorbidities_data = None
    drugs_data = None
    MedicationsView_data = None
    recommendation_medication = None
    recommendation_dosage = None   
    rec_med = None
    rec_dos = None
    meds_list = None
    last_med = None
    last_dos = None
class PatientInput(BaseModel):
    patient_id: int

@app.post("/patient_id")
def check_patient_id(patient_input: PatientInput):
    global patient_id, recommendation, patient_data, medications_data, blood_pressure_data, height_data, weight_data, lab_tests_data, comorbidities_data, drugs_data, MedicationsView_data,rec_med,rec_dos,meds_list,last_med,last_dos
    filtered_patient_df = patients_df[patients_df['PatientSourceID'].isin(patients['PatientSourceID'])]

    patient_id = patient_input.patient_id
    if patient_id in filtered_patient_df["PatientSourceID"].values:
        # If patient ID exists, fetch patient data and generate LifeStyle recommendations and Medication recommendation
        patient_data, medications_data, blood_pressure_data, height_data, weight_data, lab_tests_data, comorbidities_data, drugs_data, MedicationsView_data = fetch_patient_data(
            patients_df, medications_df, drugs_df, blood_pressure_df, height_df, weight_df, lab_tests_df, comorbidities_df, MedicationsView_df, patient_id)
        algo, trainset = test(merged_data)
        rec_med,rec_dos,meds_list = get_item_recommendations(patient_id, algo, trainset, drugs_lookup, patient_profile_matrix)
        recommendation = generate_recommendation(patient_data, medications_data, blood_pressure_data, height_data, weight_data, lab_tests_data, comorbidities_data, drugs_data)
        last_med,last_dos = get_med_dos(int(MedicationsView_data["DrugSourceID"].iloc[-1]), drugs_lookup_R)
        
        # Redirect to PatientPageNew.html with additional data
        return JSONResponse(content={"message": "Patient data retrieved", "redirect_url": "./PatientPageNew.html", "recommendation": recommendation,"Updated_Medication": rec_med,"Updated_Dosage": rec_dos,"Latest_Medication":last_med,"Latest_Dosage":last_dos,"medlist":meds_list},media_type="text/html")
    else:
        # If patient ID does not exist, return an error message
        raise HTTPException(status_code=404, detail="Patient ID not found")
        

# Define endpoint to serve data
@app.get("/patient_data")
def get_patient_data():
    response_data = {
        "height": get_first_element(height_data, "RATE", "DateTimeTaken"),
        "weight": get_first_element(weight_data, "RATE", "DateTimeTaken"),
        "age": int(patient_data["Age"]),
        "sex": patient_data["Sex"],
        "fbs": get_first_element(lab_tests_data[lab_tests_data["LabTestType"] == "FBS"], "LabTestResult", "LabTestDatetimeTaken"),
        "rbs": get_first_element(lab_tests_data[lab_tests_data["LabTestType"] == "RBS"], "LabTestResult", "LabTestDatetimeTaken"),
        "bp": get_blood_pressure(blood_pressure_data),
        "hba1c": get_first_element(lab_tests_data[lab_tests_data["LabTestType"] == "HbA1C"], "LabTestResult", "LabTestDatetimeTaken"),
        "hypertension": int(patient_data["Hypertension"]),
        "diabetes": int(patient_data["DiabetesMellitus"]),
        "patient_id": patient_id,
        "recommendation": recommendation,
        "Latest_Dosage": last_dos,
        "Latest_Medication": last_med,
        "Updated_Dosage": rec_dos,
        "Updated_Medication": rec_med,
        "medlist":(meds_list)
    }

    # Convert numpy types to native Python types
    response_data = convert_numpy_types(response_data)
    return response_data
    reset_global_variables()

def generate_pdf_with_text(data: dict):
    pdf_buffer = BytesIO()

    # Create PDF canvas
    custom_page_width = 1100  # Increase width
    custom_page_height = 1000  # Increase length
    c = canvas.Canvas(pdf_buffer, pagesize=(custom_page_width, custom_page_height))
    
    
    c.setFont("Helvetica", 10)

    # Draw text on the page
    y_position = 950  # Starting y position for text
    for key, value in data.items():
        if isinstance(value, list):
            # If value is a list, draw each element on a new line
            c.drawString(100, y_position, f"{key}:")
            y_position -= 20  # Move to the next line
            for item in value:
                c.drawString(120, y_position, item)
                y_position -= 20  # Move to the next line
        else:
            c.drawString(100, y_position, f"{key}: {value}")
            y_position -= 20  # Move to the next line

    c.save()

    pdf_buffer.seek(0)
    return pdf_buffer

# Define endpoint to serve PDF with text data
@app.get("/generate_pdf_with_text")
async def generate_pdf_with_text_endpoint():
    x =  {
        "patient_id": (patient_id),
        "height": (get_first_element(height_data, "RATE", "DateTimeTaken")*2.54),
        "weight": (get_first_element(weight_data, "RATE", "DateTimeTaken")*0.45359237),
        "age": int(patient_data["Age"]),
        "sex": patient_data["Sex"],
        "hypertension": int(patient_data["Hypertension"]),
        "diabetes": int(patient_data["DiabetesMellitus"]),
        "fbs": (get_first_element(lab_tests_data[lab_tests_data["LabTestType"] == "FBS"], "LabTestResult","LabTestDatetimeTaken")),
        "rbs": (get_first_element(lab_tests_data[lab_tests_data["LabTestType"] == "RBS"], "LabTestResult","LabTestDatetimeTaken")),
        "bp": (get_blood_pressure(blood_pressure_data)),
        "hba1c": (get_first_element(lab_tests_data[lab_tests_data["LabTestType"] == "HbA1C"], "LabTestResult","LabTestDatetimeTaken")),
        "Latest_Medication":(MedicationsView_data["DrugGenericName"].iloc[0]),
        "Latest_Dosage": (MedicationsView_data["Dosage"].iloc[0]),
        "Updated_Medication":(rec_med),
        "Updated_Dosage": (rec_dos),
        "recommendation": (recommendation)
    }
    pdf_buffer = generate_pdf_with_text(x)
    
    return StreamingResponse(pdf_buffer, media_type="application/pdf")

# Main function to run the FastAPI application
def main():
    
    uvicorn.run(app, hostrec_dos="0.0.0.0", port=8000)
    
if __name__ == "__main__":
    main()
