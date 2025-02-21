import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

def calculate_km_estimates(times, censored, ci_method='log'):
    """Calculate Kaplan-Meier estimates with detailed statistics"""
    sorted_indices = np.argsort(times)
    times = np.array(times)[sorted_indices]
    censored = np.array(censored)[sorted_indices]
    
    unique_times = np.unique(times)
    results = []
    
    n_at_risk = len(times)
    survival = 1.0
    var_sum = 0.0  # For Greenwood's formula
    
    for t in unique_times:
        mask_time = times == t
        n_events = sum(mask_time & ~censored)
        n_censored = sum(mask_time & censored)
        
        if n_events > 0:
            p = 1 - n_events/n_at_risk
            survival *= p
            var_sum += n_events / (n_at_risk * (n_at_risk - n_events))
        
        # Calculate standard error based on method
        if ci_method == 'log':
            # Log-transformed CI (default in R)
            std_err = math.sqrt(var_sum)
            if survival > 0:
                ci_lower = survival * math.exp(-1.96 * std_err)
                ci_upper = survival * math.exp(1.96 * std_err)
            else:
                ci_lower = ci_upper = 0
        else:
            # Plain CI
            std_err = survival * math.sqrt(var_sum)
            ci_lower = max(0, survival - 1.96 * std_err)
            ci_upper = min(1, survival + 1.96 * std_err)
        
        results.append({
            'time': t,
            'n_risk': n_at_risk,
            'n_event': n_events,
            'n_censored': n_censored,
            'survival': survival,
            'std_err': std_err,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })
        
        n_at_risk -= (n_events + n_censored)
    
    return pd.DataFrame(results)

[... rest of the code remains the same ...]