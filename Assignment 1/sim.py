import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon

def add_group_indicators(data, variables, n_groups=3):
    for var in variables:
        # Compute the 33rd and 67th percentiles
        q1, q2 = np.percentile(data[var], [100/n_groups, 200/n_groups])
        # Assign group: 0 if <= q1, 1 if between q1 and q2, 2 if > q2.
        data[var + "_group"] = data[var].apply(lambda x: 0 if x <= q1 else (1 if x <= q2 else 2))
    return data

def create_risk_sets(data):
    risk_sets = {}
    treated_patients = data[data['treatment'] == 1]
    
    for idx, row in treated_patients.iterrows():
        eligible_controls = data[(data['treatment'] == 0) & (data['time'] >= row['time'])]
        if not eligible_controls.empty:
            risk_sets[row['id']] = eligible_controls
    return risk_sets

def calculate_mahalanobis_distance(df, cov_matrix):
    distances = {}
    inv_cov_matrix = np.linalg.pinv(cov_matrix)  # Use pseudo-inverse

    for treated_id, controls in create_risk_sets(df).items():
        treated_row = df[df['id'] == treated_id][['age', 'severity']].values.flatten()
        
        control_distances = {}
        for _, control_row in controls.iterrows():
            control_values = control_row[['age', 'severity']].values.flatten()
            dist = mahalanobis(treated_row, control_values, inv_cov_matrix)
            control_distances[control_row['id']] = dist
        distances[treated_id] = control_distances
    return distances

def optimal_matching(mahalanobis_distances, data, penalty=100.0):
    treated_ids = list(mahalanobis_distances.keys())
    control_ids = list(set(id for distances in mahalanobis_distances.values() for id in distances))
    
    if not control_ids:
        return []  # No valid matches
    
    # Initialize cost matrix
    cost_matrix = np.full((len(treated_ids), len(control_ids)), np.inf)
    
    for i, treated_id in enumerate(treated_ids):
        # Get treated patient's row
        treated_row = data[data['id'] == treated_id].iloc[0]
        for j, control_id in enumerate(control_ids):
            base_cost = mahalanobis_distances[treated_id].get(control_id, np.inf)
            # Get control patient's row
            control_row = data[data['id'] == control_id].iloc[0]
            # Count mismatches for group indicators for each variable of interest.
            mismatch = 0
            for var in ['age_group', 'severity_group']:
                if treated_row[var] != control_row[var]:
                    mismatch += 1
            additional_cost = penalty * mismatch
            cost_matrix[i, j] = base_cost + additional_cost
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = [(treated_ids[i], control_ids[j]) 
               for i, j in zip(row_ind, col_ind) 
               if cost_matrix[i, j] != np.inf]
    
    return matches

def sensitivity_analysis(matches, data):
    biases = np.linspace(1.0, 3.0, num=5)  # Different levels of bias
    results = {}
    
    treated_scores = np.array([data.loc[t, 'severity'] for t, _ in matches])
    control_scores = np.array([data.loc[c, 'severity'] for _, c in matches])
    
    # Compute differences for matched pairs
    diff = treated_scores - control_scores
    
    # If all differences are zero, return p=1.0 for all gamma.
    if np.all(diff == 0):
        return {gamma: 1.0 for gamma in biases}
    
    # Run the Wilcoxon signed-rank test (two-sided)
    _, base_p_value = wilcoxon(diff, alternative='two-sided')
    
    for gamma in biases:
        # Apply a sensitivity adjustment (this is a simplified heuristic)
        adjusted_p_value = min(1.0, base_p_value * (1 + (gamma - 1) * 0.5))
        results[gamma] = adjusted_p_value
    
    return results

# --- Main Execution ---

# Load the sample data from CSV
data = pd.read_csv('sample_data.csv')

# Add group indicators for the covariates on which to enforce balance
data = add_group_indicators(data, variables=['age', 'severity'], n_groups=3)

# Compute covariance matrix based on the original continuous variables
cov_matrix = np.cov(data[['age', 'severity']].T)

# Execute the matching steps
mahalanobis_distances = calculate_mahalanobis_distance(data, cov_matrix)
matches = optimal_matching(mahalanobis_distances, data, penalty=100.0)
sensitivity_results = sensitivity_analysis(matches, data)

# Convert NumPy types to standard Python types before printing
cleaned_matches = [(int(t), int(c)) for t, c in matches]
cleaned_sensitivity_results = {float(g): float(p) for g, p in sensitivity_results.items()}

# Display the results
print("Matches:", cleaned_matches)
print("Sensitivity Analysis Results:", cleaned_sensitivity_results)
