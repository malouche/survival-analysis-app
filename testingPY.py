import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from scipy.stats import norm

# Sample Data
data = {
    'duration': [5, 6, 6, 2, 4, 8, 9, 7, 10, 12],
    'event_observed': [1, 0, 1, 1, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Z-score for 95% CI
z = norm.ppf(1 - 0.05 / 2)

# Initialize Fitters
kmf = KaplanMeierFitter()
naf = NelsonAalenFitter()

# Fit Kaplan-Meier and Nelson-Aalen
kmf.fit(durations=df['duration'], event_observed=df['event_observed'])
naf.fit(durations=df['duration'], event_observed=df['event_observed'])

# Extract survival and cumulative hazard
km_survival = kmf.survival_function_
event_table_km = kmf.event_table  # Get event table for variance calculation

# Manually compute Greenwood's variance for Kaplan-Meier
at_risk = event_table_km['at_risk']
observed = event_table_km['observed']
variance_km = ((observed / (at_risk * (at_risk - observed))).fillna(0)).cumsum()
variance_km = (km_survival['KM_estimate'] ** 2) * variance_km.reindex(km_survival.index, method='ffill')

# Extract cumulative hazard for Nelson-Aalen
na_cum_hazard = naf.cumulative_hazard_

# Manually compute variance for Nelson-Aalen
event_table_na = naf.event_table
variance_na = (event_table_na['observed'] / (event_table_na['at_risk'] ** 2)).cumsum()
variance_na = variance_na.reindex(na_cum_hazard.index, method='ffill')

# Calculate Survival Function from Nelson-Aalen
na_survival = np.exp(-na_cum_hazard)

# Initialize DataFrames for results
km_results = pd.DataFrame({'Time': km_survival.index, 'KM_Survival': km_survival['KM_estimate']})
na_results = pd.DataFrame({'Time': na_cum_hazard.index, 'NA_Survival': na_survival['NA_estimate']})

# --- Kaplan-Meier CIs ---
# Plain (Greenwood)
km_plain_lower = km_survival['KM_estimate'] - z * np.sqrt(variance_km)
km_plain_upper = km_survival['KM_estimate'] + z * np.sqrt(variance_km)

# Arcsin Transformation
theta_km = np.arcsin(np.sqrt(km_survival['KM_estimate']))
sd_arcsin_km = 1 / (2 * np.sqrt(len(df)))
km_arcsin_lower = np.sin(theta_km - z * sd_arcsin_km)**2
km_arcsin_upper = np.sin(theta_km + z * sd_arcsin_km)**2

# Log-Log Transformation
log_log_km = np.log(-np.log(km_survival['KM_estimate']))
se_loglog_km = np.sqrt(variance_km) / (km_survival['KM_estimate'] * np.log(km_survival['KM_estimate']))
km_loglog_lower = np.exp(-np.exp(log_log_km + z * se_loglog_km))
km_loglog_upper = np.exp(-np.exp(log_log_km - z * se_loglog_km))

# Add CIs to KM results
km_results['Plain_Lower_CI'] = km_plain_lower
km_results['Plain_Upper_CI'] = km_plain_upper
km_results['Arcsin_Lower_CI'] = km_arcsin_lower
km_results['Arcsin_Upper_CI'] = km_arcsin_upper
km_results['LogLog_Lower_CI'] = km_loglog_lower
km_results['LogLog_Upper_CI'] = km_loglog_upper

# --- Nelson-Aalen CIs ---
# Plain (Standard variance)
na_plain_lower = np.exp(-(na_cum_hazard + z * np.sqrt(variance_na).values.reshape(-1, 1)))
na_plain_upper = np.exp(-(na_cum_hazard - z * np.sqrt(variance_na).values.reshape(-1, 1)))

# Arcsin Transformation
theta_na = np.arcsin(np.sqrt(na_survival['NA_estimate']))
sd_arcsin_na = 1 / (2 * np.sqrt(len(df)))
na_arcsin_lower = np.sin(theta_na - z * sd_arcsin_na)**2
na_arcsin_upper = np.sin(theta_na + z * sd_arcsin_na)**2

# Log-Log Transformation
log_log_na = np.log(-np.log(na_survival['NA_estimate']))
se_loglog_na = np.sqrt(variance_na) / na_cum_hazard['NA_estimate']

# Log-Log Transformation for Nelson-Aalen
log_log_na = np.log(-np.log(na_survival['NA_estimate']))
se_loglog_na = np.sqrt(variance_na) / na_cum_hazard['NA_estimate']

# Corrected CI without reshaping
na_loglog_lower = np.exp(-np.exp(log_log_na + z * se_loglog_na))
na_loglog_upper = np.exp(-np.exp(log_log_na - z * se_loglog_na))


# Add CIs to NA results
na_results['Plain_Lower_CI'] = na_plain_lower['NA_estimate']
na_results['Plain_Upper_CI'] = na_plain_upper['NA_estimate']
na_results['Arcsin_Lower_CI'] = na_arcsin_lower
na_results['Arcsin_Upper_CI'] = na_arcsin_upper



na_results['LogLog_Lower_CI'] = na_loglog_lower.values
na_results['LogLog_Upper_CI'] = na_loglog_upper.values

# Display results
print("\nKaplan-Meier Estimation with CIs:\n", km_results.round(4))
print("\nNelson-Aalen Estimation with CIs:\n", na_results.round(4))
