import pandas as pd
from io import StringIO

# Option 1: Reading from a CSV file
df = pd.read_csv('main_data.csv', sep='\t')



# Create DataFrame from the multi-line string
df = pd.read_csv(StringIO(data), sep='\t')

# Clean 'chances' column by removing '%' and converting to float
df['chances'] = df['chances'].str.replace('%', '').astype(float)

# Group by 'COLLEGE' and calculate the average 'chances'
acceptance_rates = df.groupby('COLLEGE')['chances'].mean().reset_index()

# Rename 'chances' to 'Acceptance Rate'
acceptance_rates.rename(columns={'chances': 'Acceptance Rate'}, inplace=True)

# Optionally, round the Acceptance Rate to two decimal places
acceptance_rates['Acceptance Rate'] = acceptance_rates['Acceptance Rate'].round(2)

# Sort the DataFrame by 'Acceptance Rate' in descending order (optional)
acceptance_rates = acceptance_rates.sort_values(by='Acceptance Rate', ascending=False)

# Save the acceptance rates to a CSV file
acceptance_rates.to_csv('acceptance_rates.csv', index=False)

print("Acceptance rates have been calculated and saved to 'acceptance_rates.csv'.")
