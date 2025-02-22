import streamlit as st
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, NelsonAalenFitter
import plotly.graph_objects as go
import bokeh.plotting as bkp
from bokeh.models import ColumnDataSource
import plotnine as gg

# Configure the page
st.set_page_config(
    page_title="Survival Analysis App",
    layout="wide"
)

# Rest of the code remains the same until plot_survival_plotly function

def plot_survival_plotly(kmf, censored_times=None, censored_events=None):
    fig = go.Figure()
    
    # Add survival curve
    fig.add_trace(go.Scatter(
        x=kmf.timeline,
        y=kmf.survival_function_.values.flatten(),
        mode='lines',
        name='Survival Estimate',
        line=dict(color='blue')
    ))
    
    # Add confidence intervals
    if kmf.confidence_interval_ is not None:
        fig.add_trace(go.Scatter(
            x=kmf.timeline,
            y=kmf.confidence_interval_.values[:, 0],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=kmf.timeline,
            y=kmf.confidence_interval_.values[:, 1],
            mode='lines',
            fill='tonexty',
            name='95% CI',
            line=dict(width=0)
        ))
    
    # Add censored points - using 'cross' symbol instead of 'plus'
    if censored_times is not None and censored_events is not None:
        censored_mask = censored_events == 0
        censored_times = censored_times[censored_mask]
        survival_at_censored = np.interp(censored_times, 
                                       kmf.timeline,
                                       kmf.survival_function_.values.flatten())
        
        fig.add_trace(go.Scatter(
            x=censored_times,
            y=survival_at_censored,
            mode='markers',
            name='Censored',
            marker=dict(
                symbol='cross',  # Changed from 'plus' to 'cross'
                size=10,
                color='red'
            )
        ))
    
    fig.update_layout(
        title='Survival Function Estimate',
        xaxis_title='Time',
        yaxis_title='Survival Probability',
        template='plotly_white'
    )
    
    return fig

# Rest of the code remains the same...

def plot_survival_bokeh(kmf, censored_times=None, censored_events=None):
    # ... same as before ...

def plot_survival_ggplot(kmf, censored_times=None, censored_events=None):
    # ... same as before ...

def get_equivalent_r_code(method, ci_method, variables):
    # ... same as before ...

def main():
    # ... same as before ...

if __name__ == '__main__':
    main()