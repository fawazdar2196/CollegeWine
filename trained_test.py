import numpy as np
import pandas as pd
import joblib
from train import get_real_averages  # Load the function from train.py

# Load the pre-trained model and scaler
model = joblib.load('logistic_regression_best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Update the function to include EC rating in the prediction
def predict_admission(student_gpa, student_sat, college_ranking, ec_rating):
    real_averages = get_real_averages()  # Load college averages
    
    # Extract relevant college data based on college ranking
    college_data = real_averages.loc[college_ranking]
    avg_sat_college = college_data['avg_SAT']
    avg_gpa_college = college_data['avg_GPA']
    acceptance_rate = college_data['acceptance_rate']
    
    # Create a DataFrame for the student's data, now including EC rating
    student_data = pd.DataFrame({
        'GPA': [student_gpa],
        'SAT_score': [student_sat],
        'Avg_SAT_College': [avg_sat_college],
        'Avg_GPA_College': [avg_gpa_college],
        'Acceptance_Rate': [acceptance_rate],
        'EC_Rating': [ec_rating],
        'GPA_EC_Interaction': [student_gpa * ec_rating],  # Include GPA-EC interaction term
        'SAT_EC_Interaction': [student_sat * ec_rating]   # Include SAT-EC interaction term
    })

    # Scale the student data using the same scaler used during model training
    student_data_scaled = scaler.transform(student_data)
    
    # Predict the probability of admission using the logistic regression model
    probability_of_admission = model.predict_proba(student_data_scaled)[:, 1][0]
    
    return probability_of_admission * 100  

# Example student details
student_gpa = 3.93
student_sat = 1550
college_ranking = 'Stanford University'
ec_rating = 3  # Assume the student has an EC rating of 4

# Calculate the admission probability
admission_chance = predict_admission(student_gpa, student_sat, college_ranking, ec_rating)
print(f"The probability of admission is: {admission_chance:.2f}%")
