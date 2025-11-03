import pandas as pd
import numpy as np
import random

# --- Configuration ---
NUM_ROWS = 50000
OUTPUT_FILE = 'healthy_synthetic_data_v3.csv'

# --- Define possible values for categorical features ---
employment_options = ['Private', 'Public']
salary_freq_options = ['Monthly', 'Bimonthly', 'Biweekly', 'Weekly']
housing_options = ['Owned', 'Rented']
yes_no_options = ['Yes', 'No']
comaker_options = ['Friend', 'Sibling', 'Parent', 'Spouse']
income_options = ['None', 'Freelance', 'Business', 'OFW Remittance']
disaster_prep_options = ['None', 'Savings', 'Insurance', 'Community Plan']

# Define 4-level categorical options
community_role_options = ['None', 'Member', 'Leader', 'Multiple Leader']
# --- NEW: Define Paluwagan Frequency options ---
paluwagan_part_options = ['Never', 'Rarely', 'Sometimes', 'Frequently']


# --- Main Generation Logic ---
data = []
for _ in range(NUM_ROWS):
    # 1. Generate Base Features Independently
    employment_sector = random.choice(employment_options)
    employment_tenure = random.randint(6, 120)
    net_salary = random.uniform(15000, 45000)
    number_of_dependents = random.randint(0, 5)
    
    # Use new 4-level options for community role and paluwagan frequency
    has_community_role = random.choice(community_role_options)
    paluwagan_participation = random.choice(paluwagan_part_options)

    comaker_relationship = random.choice(comaker_options)
    
    # Generate comaker stats independently
    comaker_tenure = random.randint(10, 120)
    comaker_salary = random.uniform(15000, 45000)

    # 2. Calculate a Base Probability of Default
    prob_default = 0.35 # Adjusted base rate slightly

    # 3. Adjust Probability Based on Features
    # Financial Stability
    prob_default -= (employment_tenure / 12) * 0.02
    prob_default -= (net_salary - 25000) / 1000 * 0.01
    prob_default += number_of_dependents * 0.04

    # --- UPDATED: Cultural Context based on new features ---
    # Paluwagan frequency now has a graded effect
    paluwagan_freq_map = {'Never': 0.0, 'Rarely': 0.05, 'Sometimes': 0.12, 'Frequently': 0.20}
    prob_default += paluwagan_freq_map[paluwagan_participation]

    # Community role has a graded effect
    community_role_map = {'None': 0.0, 'Member': 0.05, 'Leader': 0.10, 'Multiple Leader': 0.15}
    prob_default += community_role_map[has_community_role]
    
    # Comaker strength
    comaker_strength_map = {'Friend': 0.02, 'Sibling': 0.05, 'Parent': 0.08, 'Spouse': 0.12}
    prob_default -= comaker_strength_map[comaker_relationship]
    prob_default -= (comaker_tenure / 12) * 0.005

    # Other effects
    if random.choice(housing_options) == 'Owned':
        prob_default -= 0.05

    # 4. Add Random Noise and Determine Final Target
    prob_default = np.clip(prob_default, 0.05, 0.95)
    default = 1 if np.random.rand() < prob_default else 0

    # Assemble the row
    row = {
        'Employment_Sector': employment_sector,
        'Employment_Tenure_Months': employment_tenure,
        'Net_Salary_Per_Cutoff': round(net_salary, 2),
        'Salary_Frequency': random.choice(salary_freq_options),
        'Housing_Status': random.choice(housing_options),
        'Years_at_Current_Address': round(random.uniform(0.1, 10), 1),
        'Household_Head': random.choice(yes_no_options),
        'Number_of_Dependents': number_of_dependents,
        'Comaker_Relationship': comaker_relationship,
        'Comaker_Employment_Tenure_Months': comaker_tenure,
        'Comaker_Net_Salary_Per_Cutoff': round(comaker_salary, 2),
        'Has_Community_Role': has_community_role,
        'Paluwagan_Participation': paluwagan_participation,
        'Other_Income_Source': random.choice(income_options),
        'Disaster_Preparedness': random.choice(disaster_prep_options),
        'Is_Renewing_Client': 0,
        'Default': default
    }
    data.append(row)

# --- Create and Save DataFrame ---
df = pd.DataFrame(data)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Successfully created '{OUTPUT_FILE}' with {len(df)} rows.")
print("\nFirst 5 rows of the new dataset:")
print(df.head())
print("\nDistribution of the 'Default' target variable:")
print(df['Default'].value_counts(normalize=True))
print("\nDistribution of 'Paluwagan_Participation':")
print(df['Paluwagan_Participation'].value_counts())
print("\nDistribution of 'Has_Community_Role':")
print(df['Has_Community_Role'].value_counts())
