# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

# Load the main dataset and acceptance rate from separate CSV files
df = pd.read_csv('main_data.csv')
acceptance_rates_df = pd.read_csv('college_acceptance_rates.csv')

# Merge acceptance rates into the main dataframe
df = df.merge(acceptance_rates_df, on='COLLEGE', how='left')

# Remove '%' from 'chances' and convert to float
# Ensure 'chances' is treated as string before using .str methods
df['chances'] = df['chances'].astype(str).str.replace('%', '').astype(float)

# Check for missing acceptance rates
missing_rates = df[df['Acceptance Rate'].isnull()]['COLLEGE'].unique()
if len(missing_rates) > 0:
    print("\nColleges with missing acceptance rates:")
    print(missing_rates)
    # Assign average acceptance rate
    average_acceptance_rate = np.mean(acceptance_rates_df['Acceptance Rate'])
    df['Acceptance Rate'].fillna(average_acceptance_rate, inplace=True)

# Convert acceptance rates to selectivity scores
df['Selectivity Score'] = 100 / df['Acceptance Rate']

# Features and target variable
features = ['Student GPA', 'Student SAT', 'Student EC', 'Selectivity Score']
X = df[features]
y = df['chances']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train the Gradient Boosting Regressor
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_scaled, y)

# Input from user
student_gpa = 3.9
student_sat = 1580
student_ec = 5
college_name = "Baylor University"

# Find the selectivity score for the entered college
college_selectivity = df.loc[df['COLLEGE'] == college_name, 'Selectivity Score'].values

if len(college_selectivity) == 0:
    print(f"College '{college_name}' not found in the dataset.")
else:
    # Create a DataFrame for the user input
    input_data = pd.DataFrame({
        'Student GPA': [student_gpa],
        'Student SAT': [student_sat],
        'Student EC': [student_ec],
        'Selectivity Score': [college_selectivity[0]]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict the chances of admission
    predicted_chance = model.predict(input_data_scaled).round(2)

    # Print the predicted chance
    print(f"Predicted probability of admission for {college_name}: {predicted_chance[0]}%")
