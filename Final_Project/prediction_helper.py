# backend with API
# This code will help us to predict the output based on user input using the trained models

import joblib
import pandas 

model_young = joblib.load("artifacts/model_young.joblib")
model_rest = joblib.load("artifacts/model_rest.joblib")

scaler_young = joblib.load("artifacts/scaler_young.joblib")
scaler_rest = joblib.load("artifacts/scaler_rest.joblib")

def calculate_normalized_risk(medical_history):
    risk_scores = {
                    "diabetes" : 6,
                        "heart disease": 8,
                        "high blood pressure": 6,
                        "thyroid" : 5,
                        "no disease" : 0,
                        "none" : 0
                    }
    diseases = medical_history.lower().split("&")

    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases) 

    max_score = 14
    min_score = 0

    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)  # 0 to 1

    return normalized_risk_score

def preprocess_input():
    expected_columns = [ 'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed']
    
    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2 , 'Gold' : 3}
    df = pd.DataFrame(0, columns = expected_columns, index = [0] ) 

    
    


def predict(input_dict):
    input_df = preprocess_input(input_dict)

    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction)
