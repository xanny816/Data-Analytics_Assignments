import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Step 1: Load the dummy data
data = pd.read_csv("data_censored.csv")
print("Data preview:")
print(data.head())

# Step 2: Apply Clustering on Baseline Characteristics (age, x1, x2, x3)
features = data[["age", "x1", "x2", "x3"]]
kmeans = KMeans(n_clusters=3, random_state=42)
data["cluster"] = kmeans.fit_predict(features)

# Step 3: Fit Switching Weight Models with Cluster Adjustment
switch_model_numer = smf.logit("treatment ~ age + C(cluster)", data=data).fit(disp=False)
switch_model_denom = smf.logit("treatment ~ age + x1 + x3 + C(cluster)", data=data).fit(disp=False)

data["switch_prob_numer"] = switch_model_numer.predict(data)
data["switch_prob_denom"] = switch_model_denom.predict(data)
data["switch_weight"] = data["switch_prob_numer"] / data["switch_prob_denom"]

# Step 4: Fit Censoring Weight Models with Cluster Adjustment
censor_model_numer = smf.logit("censored ~ x2 + C(cluster)", data=data).fit(disp=False)
censor_model_denom = smf.logit("censored ~ x2 + x1 + C(cluster)", data=data).fit(disp=False)

data["cens_prob_numer"] = censor_model_numer.predict(data)
data["cens_prob_denom"] = censor_model_denom.predict(data)
data["censor_weight"] = data["cens_prob_numer"] / data["cens_prob_denom"]

# Step 5: Combine weights
data["weight"] = data["switch_weight"] * data["censor_weight"]

# Step 6: Fit Outcome Model with Cluster Adjustment
outcome_model = smf.wls("outcome ~ treatment + x2 + C(cluster)", data=data, weights=data["weight"]).fit()
print("\nSimplified Outcome Model Coefficients with Clustering:")
print(outcome_model.params)

# Step 7: Expand Data for Follow-Up
followup_times = np.arange(0, 11)
expanded = pd.concat([data.assign(followup_time=t) for t in followup_times], ignore_index=True)

# Step 8: Fit a MSM Including Cluster as a Factor
msm_model = smf.wls("outcome ~ treatment + followup_time + x2 + C(cluster)", data=expanded, weights=expanded["weight"]).fit()
print("\nSimplified MSM Model Coefficients with Clustering:")
print(msm_model.params)

# Step 9: Predict and Plot Outcomes Over Follow-Up by Cluster
pred_times = np.arange(0, 11)
plt.figure(figsize=(8,6))
for cl in sorted(data["cluster"].unique()):
    cluster_data = data[data["cluster"] == cl]
    predictions = []
    for t in pred_times:
        temp = cluster_data.copy()
        temp["followup_time"] = t
        pred = msm_model.predict(temp)
        predictions.append(np.average(pred, weights=temp["weight"]))
    lower_bound = [p - 0.1 for p in predictions]
    upper_bound = [p + 0.1 for p in predictions]
    
    plt.plot(pred_times, predictions, label=f"Cluster {cl}")
    plt.plot(pred_times, lower_bound, "r--")
    plt.plot(pred_times, upper_bound, "r--")

plt.xlabel("Follow-up Time")
plt.ylabel("Survival Difference")
plt.title("Predicted Survival Difference Over Follow-up by Cluster")
plt.legend()
plt.show()