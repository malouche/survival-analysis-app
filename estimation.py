import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, NelsonAalenFitter

def estimate_km(df, alpha, ci_method):
    kmf = KaplanMeierFitter(alpha=alpha)
    kmf.fit(durations=df['time'], event_observed=df['event'])
    
    # Plain CI is used by default. For 'Arcsine' or 'Delta', add the corresponding transformations.
    timeline = kmf.survival_function_.index.values
    survival = kmf.survival_function_['KM_estimate'].values
    ci_lower = kmf.confidence_interval_['KM_estimate_lower_{}'.format(1 - alpha)].values
    ci_upper = kmf.confidence_interval_['KM_estimate_upper_{}'.format(1 - alpha)].values

    table = kmf.event_table.reset_index().rename(columns={'at_risk': 'n.risks', 'observed': 'n.events'})
    merged = pd.DataFrame({
        'time': timeline,
        'Survival': np.round(survival, 3),
        'CI': [f"{np.round(l, 3)}-{np.round(u, 3)}" for l, u in zip(ci_lower, ci_upper)]
    })
    merged = pd.merge(merged, table[['event_at', 'n.risks', 'n.events']],
                      left_on='time', right_on='event_at', how='inner').drop(columns='event_at')
    return merged, kmf

def estimate_na(df, alpha):
    naf = NelsonAalenFitter(alpha=alpha)
    naf.fit(durations=df['time'], event_observed=df['event'])
    
    timeline = naf.cumulative_hazard_.index.values
    cum_hazard = naf.cumulative_hazard_['NA_estimate'].values
    survival = np.exp(-cum_hazard)
    ci_lower = np.exp(-naf.confidence_interval_['NA_estimate_upper_{}'.format(1 - alpha)].values)
    ci_upper = np.exp(-naf.confidence_interval_['NA_estimate_lower_{}'.format(1 - alpha)].values)

    table = naf.event_table.reset_index().rename(columns={'at_risk': 'n.risks', 'observed': 'n.events'})
    merged = pd.DataFrame({
        'time': timeline,
        'Survival': np.round(survival, 3),
        'CI': [f"{np.round(l, 3)}-{np.round(u, 3)}" for l, u in zip(ci_lower, ci_upper)]
    })
    merged = pd.merge(merged, table[['event_at', 'n.risks', 'n.events']],
                      left_on='time', right_on='event_at', how='inner').drop(columns='event_at')
    return merged, naf
