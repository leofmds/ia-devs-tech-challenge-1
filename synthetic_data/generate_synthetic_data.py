import pandas as pd
import numpy as np

np.random.seed(42)

num_samples = 1000
ages = np.random.randint(18, 95, size=num_samples)
genders = np.random.choice(['male', 'female'], size=num_samples)
bmis = np.random.uniform(15, 40, size=num_samples)
children = np.random.randint(0, 5, size=num_samples)
smokers = np.random.choice(['yes', 'no'], size=num_samples)
regions = np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], size=num_samples)
costs = (
    1000 + ages * 300 + bmis * 250 + children * 500 +
    np.where(smokers == 'yes', 20000, 0) +
    np.random.normal(0, 5000, size=num_samples)
).clip(0)

data = {
    'age': ages,
    'gender': genders,
    'bmi': bmis,
    'children': children,
    'smoker': smokers,
    'region': regions,
    'cost': costs
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('synthetic_medical_costs_1.csv', index=False)
print("CSV file 'synthetic_medical_costs.csv' generated successfully.")
