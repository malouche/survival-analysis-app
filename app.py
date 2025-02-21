import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import exp, sqrt, log

st.set_page_config(
    page_title="Survival Analysis Calculator",
    layout="wide"
)

def calculate_km_estimates(times, censored, ci_method='log'):
    sorted_indices = np.argsort(times)
    times = np.array(times)[sorted_indices]
    censored = np.array(censored)[sorted_indices]
    
    unique_times = np.unique(times)
    results = []
    
    n_at_risk = len(times)
    survival = 1.0
    var_sum = 0.0
    
    for t in unique_times:
        mask_time = times == t
        n_events = sum(mask_time & ~censored)
        n_censored = sum(mask_time & censored)
        
        if n_events > 0:
            p = 1 - n_events/n_at_risk
            survival *= p
            var_sum += n_events / (n_at_risk * (n_at_risk - n_events))
        
        if ci_method == 'log':
            std_err = sqrt(var_sum)
            if survival > 0:
                ci_lower = survival * exp(-1.96 * std_err)
                ci_upper = survival * exp(1.96 * std_err)
            else:
                ci_lower = ci_upper = 0
        else:
            std_err = survival * sqrt(var_sum)
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

def plot_survival_curve(df, method):
    fig = go.Figure()
    
    times = df['time'].tolist()
    survival = df['survival'].tolist()
    
    x = [0]
    y = [1]
    
    # Create step function
    for i in range(len(times)):
        if i > 0:
            x.extend([times[i], times[i]])
            y.extend([y[-1], survival[i]])
        else:
            x.append(times[i])
            y.append(survival[i])
        
        if i < len(times) - 1:
            x.append(times[i+1])
            y.append(survival[i])
    
    # Add survival curve
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Survival Probability',
        line=dict(color='blue', width=2)
    ))
    
    # Add confidence intervals
    if all(col in df.columns for col in ['ci_lower', 'ci_upper']):
        ci_lower = df['ci_lower'].tolist()
        ci_upper = df['ci_upper'].tolist()
        
        x_ci = []
        y_ci_lower = [1]
        y_ci_upper = [1]
        
        for i in range(len(times)):
            if i > 0:
                x_ci.extend([times[i], times[i]])
                y_ci_lower.extend([y_ci_lower[-1], ci_lower[i]])
                y_ci_upper.extend([y_ci_upper[-1], ci_upper[i]])
            x_ci.append(times[i])
            y_ci_lower.append(ci_lower[i])
            y_ci_upper.append(ci_upper[i])
            
            if i < len(times) - 1:
                x_ci.append(times[i+1])
                y_ci_lower.append(ci_lower[i])
                y_ci_upper.append(ci_upper[i])
        
        fig.add_trace(go.Scatter(
            x=x_ci + x_ci[::-1],
            y=y_ci_upper + y_ci_lower[::-1],
            fill='toself',
            fillcolor='rgba(0,0,255,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI'
        ))
    
    # Add censored points
    if 'n_censored' in df.columns:
        censored_mask = df['n_censored'] > 0
        if any(censored_mask):
            fig.add_trace(go.Scatter(
                x=df.loc[censored_mask, 'time'],
                y=df.loc[censored_mask, 'survival'],
                mode='markers',
                name='Censored',
                marker=dict(
                    symbol='x',
                    size=8,
                    color='black'
                )
            ))
    
    fig.update_layout(
        title=f'{method} Survival Estimate',
        xaxis_title='Time',
        yaxis_title='Survival Probability',
        yaxis=dict(range=[0, 1.05]),
        template='plotly_white',
        showlegend=True
    )
    
    return fig

def format_results_table(df):
    display_df = df.copy()
    display_df['survival'] = display_df['survival'].map('{:.4f}'.format)
    display_df['std_err'] = display_df['std_err'].map('{:.4f}'.format)
    display_df['95% CI'] = display_df.apply(
        lambda x: f"[{x['ci_lower']:.4f}, {x['ci_upper']:.4f}]", 
        axis=1
    )
    
    return display_df[[
        'time', 'n_risk', 'n_event', 'n_censored', 
        'survival', 'std_err', '95% CI'
    ]].rename(columns={
        'time': 'Time',
        'n_risk': 'N at Risk',
        'n_event': 'N Events',
        'n_censored': 'N Censored',
        'survival': 'Survival',
        'std_err': 'Std Error'
    })

def main():
    st.title('Survival Analysis Calculator')
    
    method = st.radio(
        "Select estimation method:",
        ["Kaplan-Meier", "Nelson-Aalen"]
    )
    
    ci_method = st.radio(
        "Select confidence interval method:",
        ["log", "plain"],
        help="'log' matches R's default method, 'plain' uses linear scale"
    )
    
    st.write("### Data Entry")
    
    n = st.number_input('Enter number of observations:', min_value=1, value=3)
    
    times = []
    censored = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("##### Enter Times")
        for i in range(n):
            times.append(
                st.number_input(f'Time {i+1}:', 
                              min_value=0.0, 
                              key=f'time_{i}',
                              step=0.1)
            )
    
    with col2:
        st.write("##### Censoring Status")
        for i in range(n):
            censored.append(
                st.checkbox(f'Censored {i+1}?', 
                          key=f'cens_{i}')
            )
    
    if st.button('Calculate and Plot'):
        if not times:
            st.error("Please enter at least one time value.")
            return
        
        results_df = calculate_km_estimates(times, censored, ci_method)
        
        fig = plot_survival_curve(results_df, method)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### Results Table")
        display_df = format_results_table(results_df)
        st.dataframe(display_df, hide_index=True)

if __name__ == '__main__':
    main()