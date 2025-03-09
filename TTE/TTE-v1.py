import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Step 1: Load the dummy data (assumed saved as 'data_censored.csv')
data = pd.read_csv("data_censored.csv")
print("Data preview:")
print(data.head())

# Step 2: Fit Switching Weight Models
switch_model_numer = smf.logit("treatment ~ age", data=data).fit(disp=False)
switch_model_denom = smf.logit("treatment ~ age + x1 + x3", data=data).fit(disp=False)

data["switch_prob_numer"] = switch_model_numer.predict(data)
data["switch_prob_denom"] = switch_model_denom.predict(data)
data["switch_weight"] = data["switch_prob_numer"] / data["switch_prob_denom"]

# Step 3: Fit Censoring Weight Models
censor_model_numer = smf.logit("censored ~ x2", data=data).fit(disp=False)
censor_model_denom = smf.logit("censored ~ x2 + x1", data=data).fit(disp=False)

data["cens_prob_numer"] = censor_model_numer.predict(data)
data["cens_prob_denom"] = censor_model_denom.predict(data)
data["censor_weight"] = data["cens_prob_numer"] / data["cens_prob_denom"]

# Step 4: Combine weights
data["weight"] = data["switch_weight"] * data["censor_weight"]

# Step 5: Fit Outcome Model using Weighted Regression
outcome_model = smf.wls("outcome ~ treatment + x2", data=data, weights=data["weight"]).fit()
print("\nSimplified Outcome Model Coefficients:")
print(outcome_model.params)

# Step 6: Expand Data for Follow-Up (simulate follow-up times 0 to 10)
followup_times = np.arange(0, 11)
expanded = pd.concat([data.assign(followup_time=t) for t in followup_times], ignore_index=True)

# Step 7: Fit a Marginal Structural Model (MSM) on Expanded Data
msm_model = smf.wls("outcome ~ treatment + followup_time + x2", data=expanded, weights=expanded["weight"]).fit()
print("\nSimplified MSM Model Coefficients:")
print(msm_model.params)

# Step 8: Predict Outcomes Over Follow-Up
pred_times = np.arange(0, 11)
predictions = []
for t in pred_times:
    temp = data.copy()
    temp["followup_time"] = t
    pred = msm_model.predict(temp)
    predictions.append(np.average(pred, weights=temp["weight"]))

lower_bound = [p - 0.1 for p in predictions]
upper_bound = [p + 0.1 for p in predictions]

# Step 9: Plot the Predicted Survival Difference (similar to the R plot)
plt.figure(figsize=(8,6))
plt.plot(pred_times, predictions, label="Survival Difference", color="blue")
plt.plot(pred_times, lower_bound, "r--", label="2.5% CI")
plt.plot(pred_times, upper_bound, "r--", label="97.5% CI")
plt.xlabel("Follow-up Time")
plt.ylabel("Survival Difference")
plt.title("Predicted Survival Difference Over Follow-up")
plt.legend()
plt.show()