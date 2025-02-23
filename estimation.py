import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from scipy.stats import norm

def calculate_ci(survival, variance, alpha, method='plain'):
    """Calculate confidence intervals using different methods."""
    z = norm.ppf(1 - alpha / 2)
    
    if method.lower() == 'plain':
        lower = survival - z * np.sqrt(variance)
        upper = survival + z * np.sqrt(variance)
    elif method.lower() == 'arcsin':
        theta = np.arcsin(np.sqrt(survival))
        sd = 1 / (2 * np.sqrt(len(survival)))
        lower = np.sin(theta - z * sd)**2
        upper = np.sin(theta + z * sd)**2
    elif method.lower() == 'log-log':
        log_log = np.log(-np.log(survival))
        se_loglog = np.sqrt(variance) / (survival * np.log(survival))
        lower = np.exp(-np.exp(log_log + z * se_loglog))
        upper = np.exp(-np.exp(log_log - z * se_loglog))
    
    return lower, upper

def estimate_km(df, alpha=0.05, ci_method='plain'):
    """Estimate survival function using Kaplan-Meier with specified CI method."""
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df['time'], event_observed=df['event'])
    
    # Calculate variance using Greenwood's formula
    at_risk = kmf.event_table['at_risk']
    observed = kmf.event_table['observed']
    variance = ((observed / (at_risk * (at_risk - observed))).fillna(0)).cumsum()
    variance = (kmf.survival_function_['KM_estimate'] ** 2) * variance.reindex(kmf.survival_function_.index, method='ffill')
    
    # Calculate CIs
    lower, upper = calculate_ci(
        kmf.survival_function_['KM_estimate'].values,
        variance.values,
        alpha,
        ci_method
    )
    
    # Create results table
    table = kmf.event_table.reset_index().rename(columns={'at_risk': 'n.risks', 'observed': 'n.events'})
    results = pd.DataFrame({
        'time': kmf.survival_function_.index,
        'KM_estimate': kmf.survival_function_['KM_estimate'].round(3),
        'KM_CI_lower': lower.round(3),
        'KM_CI_upper': upper.round(3)
    })
    
    results = pd.merge(
        results,
        table[['event_at', 'n.risks', 'n.events']],
        left_on='time',
        right_on='event_at',
        how='inner'
    ).drop(columns='event_at')
    
    return results, kmf

def estimate_na(df, alpha=0.05, ci_method='plain'):
    """Estimate survival function using Nelson-Aalen with specified CI method."""
    naf = NelsonAalenFitter()
    naf.fit(durations=df['time'], event_observed=df['event'])
    
    # Calculate variance
    event_table = naf.event_table
    variance = (event_table['observed'] / (event_table['at_risk'] ** 2)).cumsum()
    variance = variance.reindex(naf.cumulative_hazard_.index, method='ffill')
    
    # Convert cumulative hazard to survival
    survival = np.exp(-naf.cumulative_hazard_['NA_estimate'])
    
    # Calculate CIs
    if ci_method.lower() == 'plain':
        lower = np.exp(-(naf.cumulative_hazard_['NA_estimate'] + 
                        norm.ppf(1 - alpha/2) * np.sqrt(variance)))
        upper = np.exp(-(naf.cumulative_hazard_['NA_estimate'] - 
                        norm.ppf(1 - alpha/2) * np.sqrt(variance)))
    else:
        lower, upper = calculate_ci(survival, variance, alpha, ci_method)
    
    # Create results table
    table = naf.event_table.reset_index().rename(columns={'at_risk': 'n.risks', 'observed': 'n.events'})
    results = pd.DataFrame({
        'time': naf.cumulative_hazard_.index,
        'NA_estimate': survival.round(3),
        'NA_CI_lower': lower.round(3),
        'NA_CI_upper': upper.round(3)
    })
    
    results = pd.merge(
        results,
        table[['event_at', 'n.risks', 'n.events']],
        left_on='time',
        right_on='event_at',
        how='inner'
    ).drop(columns='event_at')
    
    return results, naf

def combine_estimates(km_results=None, na_results=None):
    """Combine KM and NA estimates into a single table."""
    if km_results is None and na_results is None:
        return None
    
    if km_results is None:
        return na_results
    if na_results is None:
        return km_results
    
    # Merge KM and NA results
    merged = pd.merge(
        km_results,
        na_results.drop(['n.risks', 'n.events'], axis=1),
        on='time',
        how='outer'
    ).fillna('-')
    
    # Reorder columns
    cols = ['time', 'n.risks', 'n.events', 
            'KM_estimate', 'KM_CI_lower', 'KM_CI_upper',
            'NA_estimate', 'NA_CI_lower', 'NA_CI_upper']
    
    return merged[cols]
