# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from io import StringIO

# Data as a multi-line string with corrected college names enclosed in double quotes
data = '''
chances,COLLEGE,Student GPA,Student SAT,Student EC
13%,Stanford University,3.9,1550,3
8%,Stanford University,3.9,1430,3
84%,Rensselaer Polytechnic Institute,3.8,1410,3
40%,Brandeis University,3.8,1390,1
8%,Cornell University,3.9,1390,5
20%,Harvard University,4,1570,5
16%,Princeton University,3.9,1550,5
8%,Princeton University,3.7,1600,2
10%,Massachusetts Institute of Technology,4,1510,3
7%,Massachusetts Institute of Technology,3.85,1580,3
18%,Massachusetts Institute of Technology,4,1590,5
13%,Harvard University,3.85,1600,4
8%,Harvard University,3.9,1410,4
10%,Stanford University,3.8,1600,5
15%,Yale University,4,1550,5
5%,Yale University,3.85,1440,3
8%,Yale University,3.85,1600,4
19%,University of Pennsylvania,4,1600,4
14%,University of Pennsylvania,3.8,1590,5
10%,University of Pennsylvania,4,1450,3
6%,University of Pennsylvania,3.8,1390,3
3%,California Institute of Technology,3.8,1390,3
7%,California Institute of Technology,3.85,1550,3
19%,California Institute of Technology,4,1600,5
18%,Duke University,4,1600,5
6%,Duke University,3.8,1510,3
3%,Duke University,3.7,1440,5
3%,Brown University,3.7,1440,5
17%,Brown University,4,1540,5
13%,Brown University,3.9,1600,3
15%,Johns Hopkins University,3.9,1600,3
19%,Johns Hopkins University,4,1540,5
5%,Johns Hopkins University,3.7,1440,5
4%,Northwestern University,3.7,1440,5
14%,Northwestern University,4,1530,3
18%,Northwestern University,4,1590,4
17%,Columbia University,4,1590,4
7%,Columbia University,3.8,1590,4
19%,Columbia University,4,1600,5
20%,Cornell University,4,1600,5
5%,Cornell University,3.8,1470,3
7%,Cornell University,3.7,1580,3
12%,"University of Chicago",3.7,1580,3
15%,"University of Chicago",4,1540,3
5%,"University of Chicago",3.5,1600,5
14%,"University of California, Berkeley",4,1540,3
16%,"University of California, Berkeley",4,1540,5
8%,"University of California, Berkeley",3.8,1600,3
6%,"University of California, Los Angeles",3.8,1600,3
14%,"University of California, Los Angeles",4,1570,3
16%,"University of California, Los Angeles",4,1570,5
19%,Rice University,4,1570,5
13%,Rice University,3.9,1570,3
5%,Rice University,3.65,1600,3
13%,Dartmouth College,3.9,1550,3
17%,Dartmouth College,4,1600,3
6%,Dartmouth College,3.7,1510,5
18%,Vanderbilt University,4,1600,3
8%,Vanderbilt University,4,1500,1
11%,Vanderbilt University,3.8,1530,3
13%,University of Notre Dame,3.8,1530,3
23%,University of Notre Dame,4,1590,3
25%,University of Notre Dame,4,1600,5
28%,"University of Michigan--Ann Arbor",4,1600,5
21%,"University of Michigan--Ann Arbor",3.9,1510,3
17%,"University of Michigan--Ann Arbor",3.8,1450,3
8%,Georgetown University,3.8,1450,3
19%,Georgetown University,3.9,1550,5
24%,Georgetown University,4,1600,5
31%,"University of North Carolina--Chapel Hill",4,1600,5
21%,"University of North Carolina--Chapel Hill",3.85,1530,3
13%,"University of North Carolina--Chapel Hill",3.7,1590,4
15%,Carnegie Mellon University,3.7,1590,4
16%,Carnegie Mellon University,4,1520,3
22%,Carnegie Mellon University,4,1600,5
24%,Emory University,4,1600,5
19%,Emory University,4,1510,3
16%,Emory University,3.7,1590,3
11%,University of Virginia,3.7,1590,3
28%,University of Virginia,4,1500,3
33%,University of Virginia,4,1600,5
22%,Washington University in St Louis,4,1600,5
18%,Washington University in St Louis,4,1540,3
13%,Washington University in St Louis,4,1440,3
41%,"University of California--Davis",4,1440,5
21%,"University of California--Davis",3.75,1600,5
40%,"University of California--Davis",4,1300,5
24%,"University of California--San Diego",4,1300,5
18%,"University of California--San Diego",3.8,1600,5
26%,"University of California--San Diego",4,1600,3
38%,University of Florida,4,1600,3
34%,University of Florida,4,1500,3
33%,University of Florida,3.8,1550,4
21%,University of Southern California,3.8,1550,4
25%,University of Southern California,4,1550,4
29%,University of Southern California,4,1590,5
51%,University of Texas--Austin,4,1590,5
45%,University of Texas--Austin,3.9,1510,3
41%,University of Texas--Austin,3.7,1460,3
22%,University of Texas--Austin,3.4,1280,3
4%,Georgia Institute of Technology,3.4,1280,3
22%,Georgia Institute of Technology,3.9,1500,3
27%,Georgia Institute of Technology,4,1550,4
23%,"University of California--Irvine",4,1550,4
27%,"University of California--Irvine",4,1600,4
7%,"University of California--Irvine",3.7,1440,3
16%,New York University,3.7,1440,3
22%,New York University,4,1510,3
27%,New York University,4,1600,5
36%,"University of California--Santa Barbara",4,1600,5
28%,"University of California--Santa Barbara",4,1480,5
17%,"University of California--Santa Barbara",3.8,1420,3
43%,University of Illinois--Urbana-Champaign,3.8,1420,3
47%,University of Illinois--Urbana-Champaign,3.9,1490,3
55%,University of Illinois--Urbana-Champaign,4,1600,5
67%,University of Wisconsin--Madison,4,1600,5
61%,University of Wisconsin--Madison,4,1480,3
46%,University of Wisconsin--Madison,3.8,1360,3
15%,Boston College,3.8,1360,3
27%,Boston College,4,1600,2
14%,Boston College,4,1200,5
86%,Rutgers University--New Brunswick,4,1200,5
87%,Rutgers University--New Brunswick,3.8,1480,3
79%,Rutgers University--New Brunswick,3.2,1240,3
2%,Tufts University,3.2,1240,3
13%,Tufts University,3.9,1500,3
19%,Tufts University,4,1590,5
67%,University of Washington,4,1590,5
56%,University of Washington,3.7,1440,5
27%,University of Washington,3.3,1300,3
5%,Boston University,3.3,1300,3
26%,Boston University,3.8,1550,4
37%,Boston University,4,1590,5
83%,Ohio State University,4,1590,5
72%,Ohio State University,4,1460,3
48%,Ohio State University,3.4,1360,3
36%,Purdue University--Main Campus,3.4,1360,3
58%,Purdue University--Main Campus,3.8,1440,4
71%,Purdue University--Main Campus,4,1510,4
62%,University of Maryland--College Park,4,1510,4
41%,University of Maryland--College Park,3.8,1320,4
70%,University of Maryland--College Park,4,1600,3
51%,Lehigh University,4,1600,3
37%,Lehigh University,3.8,1320,3
28%,Lehigh University,3.6,1280,3
80%,Texas A&M University,3.6,1280,3
88%,Texas A&M University,4,1510,3
65%,Texas A&M University,3,1400,5
15%,University of Georgia,3,1400,5
62%,University of Georgia,3.9,1420,5
70%,University of Georgia,4,1540,5
64%,University of Rochester,4,1540,5
47%,University of Rochester,3.7,1400,5
31%,University of Rochester,3.5,1360,3
46%,Virginia Tech,3.5,1360,3
77%,Virginia Tech,4,1420,1
74%,Virginia Tech,3.8,1600,3
14%,Wake Forest University,3.8,1600,3
28%,Wake Forest University,4,1600,5
7%,Wake Forest University,3.6,1460,5
11%,Case Western Reserve University,3.6,1460,5
41%,Case Western Reserve University,4,1500,3
11%,Case Western Reserve University,3.5,1600,3
20%,Florida State University,3.5,1600,3
47%,Florida State University,4,1510,4
69%,Florida State University,4,1600,4
18%,Northeastern University,4,1600,5
13%,Northeastern University,4,1460,5
4%,Northeastern University,3.6,1540,3
91%,University of Minnesota--Twin Cities,3.6,1540,3
95%,University of Minnesota--Twin Cities,4,1520,3
77%,University of Minnesota--Twin Cities,3.4,1280,3
10%,William & Mary,3.4,1280,3
33%,William & Mary,3.85,1380,3
47%,William & Mary,4,1400,4
74%,Stony Brook University--SUNY,4,1400,4
69%,Stony Brook University--SUNY,3.8,1360,4
53%,Stony Brook University--SUNY,3.6,1260,3
65%,University of Connecticut,3.6,1260,3
39%,University of Connecticut,3.3,1200,3
90%,University of Connecticut,4,1580,4
71%,Brandeis University,4,1580,4
60%,Brandeis University,3.8,1440,4
31%,Brandeis University,3.6,1320,3
96%,Michigan State University,3.6,1320,3
89%,Michigan State University,3,1320,3
74%,Michigan State University,2.6,1040,3
27%,North Carolina State University,3,1260,3
68%,North Carolina State University,3.8,1360,3
65%,North Carolina State University,4,1260,2
81%,The Pennsylvania State University--University Park,4,1260,2
78%,The Pennsylvania State University--University Park,3.6,1380,4
83%,The Pennsylvania State University--University Park,4,1560,1
88%,Rensselaer Polytechnic Institute,3.8,1320,5
86%,Rensselaer Polytechnic Institute,3.8,1320,3
57%,Rensselaer Polytechnic Institute,3.6,1280,1
42%,Santa Clara University,3.6,1280,1
68%,Santa Clara University,3.85,1340,3
73%,Santa Clara University,3.85,1420,4
97%,"University of California--Merced",3.85,1420,4
95%,"University of California--Merced",3.4,1340,3
82%,"University of California--Merced",2.6,1140,3
67%,George Washington University,3.85,1420,3
71%,George Washington University,4,1420,4
55%,George Washington University,3.5,1580,4
63%,Syracuse University,3.5,1580,4
70%,Syracuse University,3.8,1440,3
77%,Syracuse University,4,1580,4
92%,"University of Massachusetts--Amherst",4,1580,4
87%,"University of Massachusetts--Amherst",3.7,1580,4
64%,"University of Massachusetts--Amherst",3.4,1260,3
13%,University of Miami,3.4,1260,3
26%,University of Miami,3.9,1380,4
30%,University of Miami,4,1540,4
35%,Villanova University,4,1540,4
38%,Villanova University,4,1600,5
24%,Villanova University,3.8,1460,5
51%,University of Pittsburgh,3.8,1460,5
60%,University of Pittsburgh,4,1460,3
56%,University of Pittsburgh,4,1360,3
59%,Binghamton University--SUNY,4,1360,3
50%,Binghamton University--SUNY,3.75,1360,2
37%,Binghamton University--SUNY,3.5,1520,3
92%,Indiana University--Bloomington,3.5,1520,3
94%,Indiana University--Bloomington,3.9,1280,3
66%,Indiana University--Bloomington,2.6,1080,2
52%,Indiana University--Bloomington,2.2,940,1
30%,Indiana University--Bloomington,1.6,940,1
23%,Tulane University,3.8,1480,3
27%,Tulane University,4,1500,4
33%,Tulane University,4,1600,5
88%,Colorado School of Mines,4,1600,5
82%,Colorado School of Mines,4,1400,3
74%,Colorado School of Mines,3.7,1380,3
38%,Colorado School of Mines,3.3,1220,3
37%,Pepperdine University,3.3,1220,3
31%,Pepperdine University,3.3,1220,1
46%,Pepperdine University,3.3,1300,5
75%,Pepperdine University,4,1340,3
57%,Stevens Institute of Technology,4,1340,3
53%,Stevens Institute of Technology,4,1200,3
41%,Stevens Institute of Technology,3.6,1520,4
90%,University at Buffalo--SUNY,3.6,1520,4
85%,University at Buffalo--SUNY,3.6,1200,3
60%,University at Buffalo--SUNY,2.8,1200,3
51%,"University of California--Riverside",2.8,1200,3
88%,"University of California--Riverside",3.8,1360,3
61%,"University of California--Riverside",2.8,1560,3
77%,University of Delaware,2.8,1560,3
93%,University of Delaware,3.8,1280,5
52%,University of Delaware,2.2,1100,5
93%,Rutgers University--Newark,3.7,1300,3
81%,Rutgers University--Newark,3.4,1300,3
86%,Rutgers University--Newark,3,1300,3
23%,"University of California--Santa Cruz",3,1300,3
50%,"University of California--Santa Cruz",3.5,1400,3
71%,"University of California--Santa Cruz",3.9,1550,3
'''

# Read the data into a DataFrame
df = pd.read_csv(StringIO(data))

# Remove '%' from 'chances' and convert to float
df['chances'] = df['chances'].str.replace('%', '').astype(float)

# Correct college names if necessary
df['COLLEGE'] = df['COLLEGE'].str.strip()

# Complete acceptance rates for all colleges in the dataset
acceptance_rates = {
    'Stanford University': 4.0,
    'Rensselaer Polytechnic Institute': 57.0,
    'Brandeis University': 34.0,
    'Cornell University': 11.0,
    'Harvard University': 4.0,
    'Princeton University': 4.0,
    'Massachusetts Institute of Technology': 7.0,
    'Yale University': 5.0,
    'University of Pennsylvania': 8.0,
    'California Institute of Technology': 6.0,
    'Duke University': 7.0,
    'Brown University': 6.0,
    'Johns Hopkins University': 8.0,
    'Northwestern University': 9.0,
    'Columbia University': 4.0,
    'University of Chicago': 6.0,
    'University of California, Berkeley': 14.0,
    'University of California, Los Angeles': 11.0,
    'Rice University': 9.0,
    'Dartmouth College': 7.0,
    'Vanderbilt University': 7.0,
    'University of Notre Dame': 15.0,
    'University of Michigan--Ann Arbor': 23.0,
    'Georgetown University': 12.0,
    'University of North Carolina--Chapel Hill': 19.0,
    'Carnegie Mellon University': 15.0,
    'Emory University': 13.0,
    'University of Virginia': 20.0,
    'Washington University in St Louis': 16.0,
    'University of California--Davis': 39.0,
    'University of California--San Diego': 34.0,
    'University of Florida': 31.0,
    'University of Southern California': 12.0,
    'University of Texas--Austin': 29.0,
    'Georgia Institute of Technology': 21.0,
    'University of California--Irvine': 30.0,
    'New York University': 16.0,
    'University of California--Santa Barbara': 29.0,
    'University of Illinois--Urbana-Champaign': 59.0,
    'University of Wisconsin--Madison': 54.0,
    'Boston College': 26.0,
    'Rutgers University--New Brunswick': 60.0,
    'Tufts University': 11.0,
    'University of Washington': 46.0,
    'Boston University': 19.0,
    'Ohio State University': 57.0,
    'Purdue University--Main Campus': 60.0,
    'University of Maryland--College Park': 52.0,
    'Lehigh University': 32.0,
    'Texas A&M University': 63.0,
    'University of Georgia': 40.0,
    'University of Rochester': 35.0,
    'Virginia Tech': 66.0,
    'Wake Forest University': 25.0,
    'Case Western Reserve University': 30.0,
    'Florida State University': 37.0,
    'Northeastern University': 18.0,
    'University of Minnesota--Twin Cities': 70.0,
    'William & Mary': 37.0,
    'Stony Brook University--SUNY': 48.0,
    'University of Connecticut': 56.0,
    'Michigan State University': 76.0,
    'North Carolina State University': 45.0,
    'The Pennsylvania State University--University Park': 56.0,
    'Santa Clara University': 54.0,
    'University of California--Merced': 85.0,
    'George Washington University': 41.0,
    'Syracuse University': 59.0,
    'University of Massachusetts--Amherst': 65.0,
    'University of Miami': 33.0,
    'Villanova University': 31.0,
    'University of Pittsburgh': 64.0,
    'Binghamton University--SUNY': 43.0,
    'Indiana University--Bloomington': 80.0,
    'Tulane University': 13.0,
    'Colorado School of Mines': 55.0,
    'Pepperdine University': 42.0,
    'Stevens Institute of Technology': 53.0,
    'University at Buffalo--SUNY': 70.0,
    'University of California--Riverside': 57.0,
    'University of Delaware': 66.0,
    'Rutgers University--Newark': 72.0,
    'University of California--Santa Cruz': 65.0
}

# Map colleges to acceptance rates
df['Acceptance Rate'] = df['COLLEGE'].map(acceptance_rates)

# Check for missing acceptance rates
missing_rates = df[df['Acceptance Rate'].isnull()]['COLLEGE'].unique()
if len(missing_rates) > 0:
    print("\nColleges with missing acceptance rates:")
    print(missing_rates)
    # Assign average acceptance rate
    average_acceptance_rate = np.mean(list(acceptance_rates.values()))
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

# Predict on the entire dataset
df['Predicted Chances'] = model.predict(X_scaled).round(2)

# Print the actual and predicted chances
comparison = df[['chances', 'Predicted Chances']]
print("\nActual vs Predicted Chances:")
print(comparison.to_string(index=False))

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y, df['Predicted Chances'])
rmse = np.sqrt(mse)
r2 = r2_score(y, df['Predicted Chances'])

print(f"\nModel Performance on Entire Dataset:")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")