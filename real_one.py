from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import sys
import os 
import numpy as np
import django
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from decimal import Decimal
# Moving up and adding api module to current path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'CollegeAid.settings')
django.setup()
from api.models import SeedUniData

num_samples = 100000

def get_real_averages():
    # Query the database
    colleges = SeedUniData.objects.all().values('ranking', 'avg_SAT', 'avg_GPA', 'acceptance_rate')
    
    # Convert the queryset to a DataFrame
    df = pd.DataFrame(colleges)
    
    # Set 'college_id' as the index
    df.set_index('ranking', inplace=True)
    
    return df

real_averages = get_real_averages()
real_averages = real_averages[~real_averages.index.duplicated(keep='first')]

# Creating synthetic applicants
student_college_ids = np.random.choice(real_averages.index, num_samples)
gpa = np.random.uniform(3.4, 4.0, num_samples)
sat_score = np.random.uniform(1250, 1600, num_samples)

student_averages = real_averages.loc[student_college_ids]
avg_sat_college = student_averages['avg_SAT'].values
avg_gpa_college = student_averages['avg_GPA'].values
acceptance_rates = student_averages['acceptance_rate'].values

#ground truth algo

def admitted_prob(gpa, sat_score, avg_gpa_college, avg_sat_college):
    # Convert numpy types to float before using Decimal
    gpa_diff = Decimal(float(gpa)) - Decimal(float(avg_gpa_college))
    sat_diff = Decimal(float(sat_score)) - Decimal(float(avg_sat_college))

    gpa_weight = 1.25
    sat_weight = 1.0

    prob = 1 / (1 + np.exp(-(gpa_weight * float(gpa_diff) + sat_weight * float(sat_diff))))

    return int(prob >= 0.5)

admitted = [
    admitted_prob(gpa[i], sat_score[i], avg_gpa_college[i], avg_sat_college[i])
    for i, acc_rate in enumerate(acceptance_rates)
]

data = pd.DataFrame({
    'GPA': gpa,
    'SAT_score': sat_score,
    'Avg_SAT_College': avg_sat_college,
    'Avg_GPA_College': avg_gpa_college,
    'Acceptance_Rate': acceptance_rates,
    'Admitted': admitted
})

# Define features and target
X = data[['GPA', 'SAT_score', 'Avg_SAT_College', 'Avg_GPA_College', 'Acceptance_Rate']]
Y = data['Admitted']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#model
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Hyperparameter tuning


# Train with best parameters
best_params = {
    'max_depth': None, 
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'n_estimators': 50
}
 
best_model = RandomForestClassifier(**best_params)
best_model.fit(X_train, Y_train)
# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(Y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(Y_test, y_pred))

# Example student data
student_gpa = 3.85
student_sat = 1300
target_college_avg_gpa = 4.0
target_college_avg_sat = 1350
acceptance_rate = 0.3  # Example acceptance rate

# Prepare the input data as a DataFrame with column names matching those used in training
student_data = pd.DataFrame({
    'GPA': [student_gpa],
    'SAT_score': [student_sat],
    'Avg_SAT_College': [target_college_avg_sat],
    'Avg_GPA_College': [target_college_avg_gpa],
    'Acceptance_Rate': [acceptance_rate]
})

# Get probability predictions
probabilities = best_model.predict_proba(student_data)

# Extract the probability of being admitted (usually class 1)
admission_probability = probabilities[0][1]

print(f"The probability of being admitted is: {admission_probability:.2f}")