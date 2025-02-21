import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

st.set_page_config(
    page_title="Survival Analysis Calculator",
    layout="wide"
)

def calculate_km_estimates(times, censored, ci_method='log'):
    sorted_indices = np.argsort(times)
    times = np.array(times)[sorted_indices]
    censored = np.array(censored)[sorted_indices]
    
    # First, count how many events at each unique time
    unique_times = np.unique(times)
    results = []
    
    # Start with all subjects
    n_at_risk = len(times)
    survival = 1.0
    var_sum = 0.0
    
    # For each time point
    for t in unique_times:
        # Count events and censored at this time
        mask_at_time = times == t
        n_events = sum(mask_at_time & ~censored)
        n_censored = sum(mask_at_time & censored)
        
        # Update survival estimate
        if n_events > 0:
            p = 1 - n_events/n_at_risk
            survival *= p
            var_sum += n_events / (n_at_risk * (n_at_risk - n_events))
        
        # Calculate standard error and CI
        if ci_method == 'log':
            std_err = math.sqrt(var_sum)
            if survival > 0:
                ci_lower = survival * math.exp(-1.96 * std_err)
                ci_upper = survival * math.exp(1.96 * std_err)
            else:
                ci_lower = ci_upper = 0
                std_err = float('inf')
        else:
            std_err = survival * math.sqrt(var_sum)
            ci_lower = max(0, survival - 1.96 * std_err)
            ci_upper = min(1, survival + 1.96 * std_err)
            
            if survival == 0:
                std_err = float('inf')
        
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
        
        # Update n_at_risk for next time point
        n_at_risk -= (n_events + n_censored)
    
    return pd.DataFrame(results)

def plot_survival_curve(df, method):
    fig = go.Figure()
    
    # Create step function coordinates
    x_steps = [0]  # Start at time 0
    y_steps = [1]  # Start with survival = 1
    
    # Add steps for each time point
    for i in range(len(df)):
        current_time = df['time'].iloc[i]
        current_surv = df['survival'].iloc[i]
        
        # Add vertical line at current time
        x_steps.append(current_time)
        y_steps.append(y_steps[-1])  # Use previous survival until the event
        
        # Add horizontal line at new survival level
        x_steps.append(current_time)
        y_steps.append(current_surv)
    
    # Add confidence intervals
    if all(col in df.columns for col in ['ci_lower', 'ci_upper']):
        x_ci = []
        y_ci_upper = []
        y_ci_lower = []
        
        # Start at time 0
        x_ci.extend([0])
        y_ci_upper.extend([1])
        y_ci_lower.extend([1])
        
        # Add steps for CIs
        for i in range(len(df)):
            current_time = df['time'].iloc[i]
            current_upper = df['ci_upper'].iloc[i]
            current_lower = df['ci_lower'].iloc[i]
            
            # Add vertical lines
            x_ci.extend([current_time, current_time])
            y_ci_upper.extend([y_ci_upper[-1], current_upper])
            y_ci_lower.extend([y_ci_lower[-1], current_lower])
        
        # Add CI ribbon
        fig.add_trace(go.Scatter(
            x=x_ci + x_ci[::-1],
            y=y_ci_upper + y_ci_lower[::-1],
            fill='toself',
            fillcolor='rgba(0,0,255,0.1)',
            line=dict(width=0),
            showlegend=True,
            name='95% CI'
        ))
    
    # Add main survival curve
    fig.add_trace(go.Scatter(
        x=x_steps,
        y=y_steps,
        mode='lines',
        name='Survival Probability',
        line=dict(color='blue', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{method} Survival Estimate',
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            title='Time',
            zeroline=True,
            gridwidth=1,
            title_standoff=15
        ),
        yaxis=dict(
            title='Survival Probability',
            range=[0, 1.05],
            zeroline=True,
            gridwidth=1,
            title_standoff=15
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

[... rest of the code remains the same ...]