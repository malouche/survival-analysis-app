import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import exp, sqrt, log

def plot_survival_curve(df, method):
    """Create step function plot for survival curve"""
    fig = go.Figure()
    
    # Create step function
    times = df['time'].tolist()
    survival = df['survival'].tolist()
    ci_lower = df['ci_lower'].tolist()
    ci_upper = df['ci_upper'].tolist()
    
    # Create step function points
    x_steps = [0]  # Start at 0
    y_steps = [1]  # Start at 1
    y_lower_steps = [1]
    y_upper_steps = [1]
    
    # Add all time points
    for i in range(len(times)):
        # Current point
        current_time = times[i]
        current_surv = survival[i]
        current_lower = ci_lower[i]
        current_upper = ci_upper[i]
        
        # Add vertical line at event
        x_steps.extend([current_time, current_time])
        y_steps.extend([y_steps[-1], current_surv])
        y_lower_steps.extend([y_lower_steps[-1], current_lower])
        y_upper_steps.extend([y_upper_steps[-1], current_upper])
        
        # Add horizontal line if not last point
        if i < len(times) - 1:
            next_time = times[i + 1]
            x_steps.append(next_time)
            y_steps.append(current_surv)
            y_lower_steps.append(current_lower)
            y_upper_steps.append(current_upper)
    
    # Add main survival curve
    fig.add_trace(go.Scatter(
        x=x_steps,
        y=y_steps,
        mode='lines',
        name='Survival',
        line=dict(color='blue', width=2)
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=x_steps + x_steps[::-1],
        y=y_upper_steps + y_lower_steps[::-1],
        fill='toself',
        fillcolor='rgba(0,0,255,0.1)',
        line=dict(width=0),
        name='95% CI',
        showlegend=True
    ))
    
    # Add censoring marks
    censored_times = df[df['n_censored'] > 0]['time'].tolist()
    censored_survival = df[df['n_censored'] > 0]['survival'].tolist()
    
    if censored_times:
        fig.add_trace(go.Scatter(
            x=censored_times,
            y=censored_survival,
            mode='markers',
            name='Censored',
            marker=dict(
                symbol='x',
                size=8,
                color='black',
                line=dict(width=2)
            )
        ))
    
    # Update layout
    fig.update_layout(
        title=f'{method} Survival Estimate',
        xaxis_title='Time',
        yaxis_title='Survival Probability',
        yaxis=dict(range=[0, 1.05]),
        template='plotly_white',
        showlegend=True
    )
    
    return fig

[... rest of your code remains the same ...]