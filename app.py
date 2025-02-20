import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import exp, sqrt

st.set_page_config(
    page_title="Survival Analysis Calculator",
    layout="wide"
)

def calculate_km_estimates(times, censored):
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
        # Count events and censored at this time
        mask_time = times == t
        n_events = sum(mask_time & ~censored)
        n_censored = sum(mask_time & censored)
        
        if n_events > 0:
            p = 1 - n_events/n_at_risk
            survival *= p
            var_sum += n_events / (n_at_risk * (n_at_risk - n_events))
        
        # Calculate standard error using Greenwood's formula
        std_err = survival * sqrt(var_sum) if survival > 0 else 0
        
        # Calculate confidence intervals
        z = 1.96  # 95% CI
        ci_lower = max(0, survival - z * std_err)
        ci_upper = min(1, survival + z * std_err)
        
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

def calculate_na_estimates(times, censored):
    """Calculate Nelson-Aalen estimates with detailed statistics"""
    sorted_indices = np.argsort(times)
    times = np.array(times)[sorted_indices]
    censored = np.array(censored)[sorted_indices]
    
    unique_times = np.unique(times)
    results = []
    
    n_at_risk = len(times)
    cumulative_hazard = 0.0
    var_sum = 0.0
    
    for t in unique_times:
        # Count events and censored at this time
        mask_time = times == t
        n_events = sum(mask_time & ~censored)
        n_censored = sum(mask_time & censored)
        
        if n_events > 0:
            cumulative_hazard += n_events/n_at_risk
            var_sum += n_events / (n_at_risk * n_at_risk)
        
        survival = exp(-cumulative_hazard)
        
        # Calculate standard error
        std_err = survival * sqrt(var_sum)
        
        # Calculate confidence intervals
        z = 1.96  # 95% CI
        ci_lower = max(0, survival - z * std_err)
        ci_upper = min(1, survival + z * std_err)
        
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
    
    # Add survival curve
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['survival'],
        mode='lines+markers',
        name='Survival Probability',
        line=dict(color='blue', width=2)
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=df['time'].tolist() + df['time'].tolist()[::-1],
        y=df['ci_upper'].tolist() + df['ci_lower'].tolist()[::-1],
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

def generate_latex_formula(df, method):
    latex = r"S(t) = \begin{cases}"
    
    for i, row in df.iterrows():
        if i == 0:
            latex += f"1, & t < {row['time']:.2f} \\\\"
        latex += f"{row['survival']:.3f}, & {row['time']:.2f} \leq t < "
        if i < len(df) - 1:
            latex += f"{df.iloc[i+1]['time']:.2f} \\\\"
        else:
            latex += r"\infty \\"
    
    latex += r"\end{cases}"
    
    return latex

def format_results_table(df):
    """Format the results table for display"""
    display_df = df.copy()
    display_df['survival'] = display_df['survival'].map('{:.4f}'.format)
    display_df['std_err'] = display_df['std_err'].map('{:.4f}'.format)
    display_df['95% CI'] = display_df.apply(
        lambda x: f"[{x['ci_lower']:.4f}, {x['ci_upper']:.4f}]", 
        axis=1
    )
    
    # Select and rename columns
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
                              key=f'time_{i}')
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
            
        # Calculate estimates based on selected method
        if method == "Kaplan-Meier":
            results_df = calculate_km_estimates(times, censored)
        else:
            results_df = calculate_na_estimates(times, censored)
        
        # Create survival plot
        fig = plot_survival_curve(results_df, method)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display LaTeX formula
        st.write("### Survival Function Formula")
        latex_formula = generate_latex_formula(results_df, method)
        st.latex(latex_formula)
        
        # Display detailed results table
        st.write("### Results Table")
        st.write(f"{method} Survival Estimates")
        display_df = format_results_table(results_df)
        st.dataframe(display_df, hide_index=True)

if __name__ == '__main__':
    main()