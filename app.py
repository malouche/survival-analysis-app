"""
Survival Analysis App with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, NelsonAalenFitter

# Configure the page
st.set_page_config(
    page_title="Survival Analysis",
    layout="wide",
    page_icon="📊"
)

[previous code remains the same until the plot_survival_curve function]

def plot_survival_curve(df, time_col, censored_col, method='km', ci_method='plain', alpha=0.05):
    """Plot survival curve using the selected method"""
    if method == 'km':
        kmf = KaplanMeierFitter()
        kmf.fit(df[time_col], df[censored_col], alpha=alpha)
        
        fig = go.Figure()
        
        # Add survival curve
        fig.add_trace(go.Scatter(
            x=kmf.timeline,
            y=kmf.survival_function_.values.flatten(),
            mode='lines',
            name='Survival Estimate',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=kmf.timeline,
            y=kmf.confidence_interval_.values[:, 0],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name='Lower CI'
        ))
        
        fig.add_trace(go.Scatter(
            x=kmf.timeline,
            y=kmf.confidence_interval_.values[:, 1],
            mode='lines',
            fill='tonexty',
            line=dict(width=0),
            name=f'{int((1-alpha)*100)}% CI'
        ))
        
        # Add censored points
        censored_times = df[df[censored_col] == 0][time_col]
        if len(censored_times) > 0:
            censored_survival = [kmf.survival_function_.loc[kmf.timeline <= t].iloc[-1] 
                               if len(kmf.survival_function_.loc[kmf.timeline <= t]) > 0 
                               else 1.0 
                               for t in censored_times]
            
            fig.add_trace(go.Scatter(
                x=censored_times,
                y=censored_survival,
                mode='markers',
                name='Censored',
                marker=dict(
                    symbol='cross',  # Changed from 'plus' to 'cross'
                    size=8,
                    color='black',
                    line=dict(width=2)
                )
            ))
        
    else:  # Nelson-Aalen
        naf = NelsonAalenFitter()
        naf.fit(df[time_col], df[censored_col], alpha=alpha)
        
        fig = go.Figure()
        
        # Add cumulative hazard curve
        fig.add_trace(go.Scatter(
            x=naf.timeline,
            y=naf.cumulative_hazard_.values.flatten(),
            mode='lines',
            name='Cumulative Hazard',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=naf.timeline,
            y=naf.confidence_interval_.values[:, 0],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=naf.timeline,
            y=naf.confidence_interval_.values[:, 1],
            mode='lines',
            fill='tonexty',
            line=dict(width=0),
            name=f'{int((1-alpha)*100)}% CI'
        ))
    
    # Update layout
    fig.update_layout(
        title='Survival Analysis',
        xaxis_title=time_col,
        yaxis_title='Survival Probability' if method == 'km' else 'Cumulative Hazard',
        template='plotly_white',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

[rest of the code remains the same]
