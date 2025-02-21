import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import exp, sqrt

def plot_survival_curve(df, method):
    fig = go.Figure()
    
    # Create step function by duplicating points
    x_steps = []
    y_steps = []
    y_upper_steps = []
    y_lower_steps = []
    
    # Start at time 0
    x_steps.append(0)
    y_steps.append(1)
    y_upper_steps.append(1)
    y_lower_steps.append(1)
    
    # Add steps for each time point
    for i in range(len(df)):
        if i > 0:
            # Add vertical line at change point
            x_steps.append(df['time'].iloc[i])
            y_steps.append(df['survival'].iloc[i-1])
            y_upper_steps.append(df['ci_upper'].iloc[i-1])
            y_lower_steps.append(df['ci_lower'].iloc[i-1])
        
        # Add horizontal line
        x_steps.append(df['time'].iloc[i])
        y_steps.append(df['survival'].iloc[i])
        y_upper_steps.append(df['ci_upper'].iloc[i])
        y_lower_steps.append(df['ci_lower'].iloc[i])
    
    # Add survival curve
    fig.add_trace(go.Scatter(
        x=x_steps,
        y=y_steps,
        mode='lines',
        name='Survival Probability',
        line=dict(color='blue', width=2)
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=x_steps + x_steps[::-1],
        y=y_upper_steps + y_lower_steps[::-1],
        fill='toself',
        fillcolor='rgba(0,0,255,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI'
    ))
    
    fig.update_layout(
        title=f'{method} Survival Estimate',
        xaxis_title='Time',
        yaxis_title='Survival Probability',
        yaxis=dict(range=[0, 1.1]),
        template='plotly_white',
        showlegend=True
    )
    
    return fig

def generate_r_code(times, censored, method):
    """Generate equivalent R code for the analysis"""
    # Create data vectors as strings
    times_str = ", ".join(map(str, times))
    censored_str = ", ".join(map(str, [int(not x) for x in censored]))  # R uses 1 for event, 0 for censored
    
    r_code = f"""# Load required libraries
library(survival)
library(survminer)

# Create data frame
data <- data.frame(
    time = c({times_str}),
    status = c({censored_str})  # 1 = event, 0 = censored
)

# Fit survival curve
fit <- survfit(Surv(time, status) ~ 1, 
               data = data,
               type = {"'fleming-harrington'" if method == "Nelson-Aalen" else "'kaplan-meier'"})

# Print summary
print(summary(fit))

# Create plot
ggsurvplot(fit,
           data = data,
           conf.int = TRUE,
           risk.table = TRUE,
           xlab = "Time",
           ylab = "Survival Probability",
           title = "{method} Survival Estimate")
"""
    return r_code

[... rest of the code remains the same ...]