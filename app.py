import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC

# Load the trained model
svc = pickle.load(open('svc.pkl', 'rb'))

# Load additional datasets for recommendations
symptoms = pd.read_csv("symptoms_complete.csv")
causes = pd.read_csv("causes_complete.csv")
medications = pd.read_csv('medications_complete.csv')
diets = pd.read_csv("diet_complete.csv")
workout = pd.read_csv("workout_complete.csv")
precautions = pd.read_csv("precautions_complete.csv")

# Symptom dictionary (from your notebook)
symptoms_dict = {
    'Abdominal cramps': 0, 'Abdominal discomfort': 1, 'Abdominal pain': 2, 'Acne': 3, 'Aura': 4, 'Balance issues': 5, 'Balance problems': 6, 
    'Blackheads': 7, 'Bleeding': 8, 'Blisters': 9, 'Bloating': 10, 'Blurred vision': 11, 'Body aches': 12, 'Breathing difficulty': 13, 
    'Breathlessness': 14, 'Burning sensation': 15, 'Chest discomfort': 16, 'Chest pain': 17, 'Chest tightness': 18, 'Chills': 19, 'Cloudy urine': 20, 
    'Cold intolerance': 21, 'Confusion': 22, 'Cough': 23, 'Coughing': 24, 'Cracks': 25, 'Dark urine': 26, 'Daytime sleepiness': 27, 'Depression': 28, 
    'Diarrhea': 29, 'Difficulty swallowing': 30, 'Discharge': 31, 'Discomfort': 32, 'Dizziness': 33, 'Drooling': 34, 'Dry eyes': 35, 'Excess hair growth': 36, 
    'Fatigue': 37, 'Fever': 38, 'Flaking': 39, 'Food aversions': 40, 'Frequent infections': 41, 'Frequent urination': 42, 'Hair thinning': 43, 
    'Halos around lights': 44, 'Headache': 45, 'Headaches': 46, 'Heartburn': 47, 'Heavy periods': 48, 'Hunger': 49, 'Indigestion': 50, 'Inflammation': 51, 
    'Irritability': 52, 'Itching': 53, 'Itchy eyes': 54, 'Jaundice': 55, 'Joint pain': 56, 'Joint swelling': 57, 'Joint/muscle pain': 58, 'Leg pain': 59, 
    'Limited mobility': 60, 'Limited movement': 61, 'Loss of appetite': 62, 'Loss of balance': 63, 'Loss of consciousness': 64, 'Loss of interest': 65, 
    'Loss of taste': 66, 'Loss of taste/smell': 67, 'Lung infections': 68, 'Memory issues': 69, 'Mood swings': 70, 'Muscle jerks': 71, 'Muscle pain': 72, 
    'Muscle weakness': 73, 'Nausea': 74, 'Night sweats': 75, 'Nosebleeds': 76, 'Numbness': 77, 'Numbness in limbs': 78, 'Pain': 79, 'Painful intercourse': 80, 
    'Painful swallowing': 81, 'Pale complexion': 82, 'Pale skin': 83, 'Palpitations': 84, 'Pelvic discomfort': 85, 'Poor growth': 86, 'Pressure around head': 87, 
    'Prolonged bleeding': 88, 'Rapid heartbeat': 89, 'Rash': 90, 'Redness': 91, 'Reduced motion': 92, 'Regurgitation': 93, 'Runny nose': 94, 'Scaling skin': 95, 
    'Sensitivity to light': 96, 'Severe headache': 97, 'Shooting leg pain': 98, 'Shortness of breath': 99, 'Skin crusting': 100, 'Skin inflammation': 101, 
    'Skin irritation': 102, 'Skin rash': 103, 'Sleep disturbances': 104, 'Sleep issues': 105, 'Slow movements': 106, 'Slurred speech': 107, 'Sore throat': 108, 
    'Spinning sensation': 109, 'Sputum production': 110, 'Stiffness': 111, 'Sweating': 112, 'Swelling': 113, 'Swollen glands': 114, 'Swollen lymph nodes': 115, 
    'Swollen tonsils': 116, 'Tearing': 117, 'Tingling': 118, 'Trembling': 119, 'Upper abdominal pain': 120, 'Urgency': 121, 'Visible veins': 122, 
    'Vision problems': 123, 'Vomiting': 124, 'Vomiting after coughing': 125, 'Warmth': 126, 'Weakness': 127, 'Weight gain': 128, 'Weight loss': 129, 
    'Whiteheads': 130, 'Xanthomas': 131, 'Abdominal pain': 132, 'Back pain': 133, 'Bladder pain': 134, 'Bloating': 135, 'Burning stomach pain': 136, 
    'Burning urination': 137, 'Chest pain': 138, 'Chronic cough': 139, 'Cough': 140, 'Diarrhea': 141, 'Dizziness': 142, 'Dull headache': 143, 'Easy bruising': 144, 
    'Eye pain': 145, 'Eye strain': 146, 'Facial drooping': 147, 'Fatigue': 148, 'Fever': 149, 'Headaches': 150, 'Heartburn': 151, 'Heavy sweating': 152, 
    'High cholesterol': 153, 'High fever': 154, 'Hives': 155, 'Increased thirst': 156, 'Irregular periods': 157, 'Itching': 158, 'Itchy rash': 159, 'Jaundice': 160, 
    'Joint pain': 161, 'Joint stiffness': 162, 'Leg pain': 163, 'Leg swelling': 164, 'Loud snoring': 165, 'Lower abdominal cramps': 166, 'Lower back pain': 167, 
    'Lower right abdominal pain': 168, 'Muscle stiffness': 169, 'Nausea': 170, 'Neck pain': 171, 'Neck stiffness': 172, 'Nervousness': 173, 'Numbness': 174, 
    'Pain during bowel movements': 175, 'Painful rash': 176, 'Panic attacks': 177, 'Pelvic pain': 178, 'Persistent cough': 179, 'Persistent sadness': 180, 
    'Pimples': 181, 'Rash': 182, 'Red patches': 183, 'Red sores': 184, 'Red, itchy skin': 185, 'Redness in eye': 186, 'Restlessness': 187, 'Right upper abdominal pain': 188, 
    'Seizures': 189, 'Severe cough': 190, 'Shakiness': 191, 'Sneezing': 192, 'Sore throat': 193, 'Soreness': 194, 'Sudden weakness': 195, 'Swollen lymph nodes': 196, 
    'Throbbing headache': 197, 'Tremors': 198, 'Weight changes': 199, 'Wheezing': 200, 'Widespread pain': 201, 'Yellowing of skin': 202, 'Red, itchy rash': 204, 'Cracking': 205
}

# Disease list (from your notebook)
diseases_list = {
    0: 'AIDS', 1: 'Acne', 2: 'Alcoholic Hepatitis', 3: 'Allergic Reaction', 4: 'Allergy', 5: 'Anxiety Disorder', 6: 'Appendicitis', 
    7: 'Arthritis', 8: 'Asthma', 9: 'Bell’s Palsy', 10: 'Bronchial Asthma', 11: 'Bronchitis (Acute)', 12: 'COVID-19', 
    13: 'Celiac Disease', 14: 'Cervical Spondylosis', 15: 'Chickenpox', 16: 'Cholecystitis', 17: 'Chronic Bronchitis', 
    18: 'Chronic Cholestasis', 19: 'Chronic Fatigue Syndrome', 20: 'Common Cold', 21: 'Conjunctivitis', 22: 'Coronary Artery Disease', 
    23: 'Crohn’s Disease', 24: 'Cystic Fibrosis', 25: 'Deep Vein Thrombosis', 26: 'Dengue', 27: 'Dermatitis', 28: 'Diabetes', 
    29: 'Dimorphic Hemorrhoids (Piles)', 30: 'Diverticulitis', 31: 'Drug Reaction', 32: 'Eczema', 33: 'Endometriosis', 
    34: 'Epilepsy', 35: 'Fibromyalgia', 36: 'Fungal Infection', 37: 'GERD', 38: 'Gallstones', 39: 'Gastritis', 
    40: 'Gastroenteritis', 41: 'Glaucoma', 42: 'Gout', 43: 'Heart Attack', 44: 'Heat Exhaustion', 45: 'Hemophilia', 
    46: 'Hepatitis A', 47: 'Hepatitis B', 48: 'Hepatitis C', 49: 'Hepatitis D', 50: 'Hepatitis E', 51: 'Herniated Disc', 
    52: 'Herpes Zoster (Shingles)', 53: 'Hodgkin’s Lymphoma', 54: 'Hyperlipidemia', 55: 'Hypertension', 56: 'Hyperthyroidism', 
    57: 'Hypoglycemia', 58: 'Hypothyroidism', 59: 'Impetigo', 60: 'Indigestion', 61: 'Infectious Mononucleosis', 62: 'Influenza', 
    63: 'Interstitial Cystitis', 64: 'Iron Deficiency', 65: 'Iron Deficiency Anemia', 66: 'Jaundice', 67: 'Lupus (SLE)', 
    68: 'Major Depressive Disorder', 69: 'Malaria', 70: 'Measles', 71: 'Meningitis (Bacterial)', 72: 'Menstrual Cramps', 
    73: 'Migraine', 74: 'Morning Sickness', 75: 'Motion Sickness', 76: 'Multiple Sclerosis', 77: 'Muscle Overuse', 
    78: 'Muscle Strain', 79: 'Osteoarthritis', 80: 'Panic Disorder', 81: 'Paralysis (Brain Hemorrhage)', 82: 'Parkinson’s Disease', 
    83: 'Peptic Ulcer Disease', 84: 'Pneumonia', 85: 'Polycystic Ovary Syndrome (PCOS)', 86: 'Psoriasis', 87: 'Respiratory Infection', 
    88: 'Rheumatoid Arthritis', 89: 'Sciatica', 90: 'Sleep Apnea', 91: 'Strep Throat', 92: 'Tension Headache', 93: 'Thyroid Disorder', 
    94: 'Tonsillitis', 95: 'Tuberculosis', 96: 'Typhoid', 97: 'Urinary Tract Infection', 98: 'Varicose Veins', 
    99: 'Vertigo (Paroxysmal Positional Vertigo)', 100: 'Vision Fatigue', 101: 'Whooping Cough (Pertussis)'
}

# Helper function to fetch recommendations
def helper(dis):
    cau = causes[causes['Disease'] == dis]['Causes'].values[0] if not causes[causes['Disease'] == dis].empty else "Not available"
    pre = precautions[precautions['Disease'] == dis][['Precaution1', 'Precaution2', 'Precaution3', 'Precaution4']].values[0] if not precautions[precautions['Disease'] == dis].empty else ["Not available"]
    med = medications[medications['Disease'] == dis]['Medication'].values if not medications[medications['Disease'] == dis].empty else ["Not available"]
    die = diets[diets['Disease'] == dis]['Diets'].values if not diets[diets['Disease'] == dis].empty else ["Not available"]
    wrkout = workout[workout['Disease'] == dis]['Workout'].values if not workout[workout['Disease'] == dis].empty else ["Not available"]
    return cau, pre, med, die, wrkout

# Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(206)  # Adjust to match feature count
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Streamlit app layout
st.title("Personalized Healthcare Recommendation System")
st.write("Enter your symptoms below to get a diagnosis and personalized recommendations.")

# Dynamic symptom input using multiselect
all_symptoms = list(symptoms_dict.keys())
selected_symptoms = st.multiselect("Select your symptoms:", all_symptoms)

# Button to trigger diagnosis
if st.button("Diagnose"):
    if not selected_symptoms:
        st.error("Please select at least one symptom.")
    else:
        with st.spinner("Diagnosing..."):
            # Get prediction
            predicted_disease = get_predicted_value(selected_symptoms)
            cau, pre, med, die, wrkout = helper(predicted_disease)

            # Display results dynamically
            st.subheader("Diagnosis Results")
            st.success(f"**Predicted Disease:** {predicted_disease}")

            st.write("### Causes")
            st.write(cau)

            st.write("### Precautions")
            for i, precaution in enumerate(pre, 1):
                st.write(f"{i}. {precaution}")

            st.write("### Medications")
            for i, medication in enumerate(med, 1):
                st.write(f"{i}. {medication}")

            st.write("### Recommended Diets")
            for i, diet in enumerate(die, 1):
                st.write(f"{i}. {diet}")

            st.write("### Recommended Workouts")
            for i, workout_item in enumerate(wrkout, 1):
                st.write(f"{i}. {workout_item}")

# Footer
st.write("---")
st.write("Developed by Mahek Qureshi  & Charul Patel | Built with Streamlit.")