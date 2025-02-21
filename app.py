import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

st.set_page_config(
    page_title="Survival Analysis Calculator",
    layout="wide"
)

# Custom CSS to control input width and alignment
st.markdown("""
    <style>
    .stNumberInput div {width: 150px;}
    .row-widget.stCheckbox {padding-top: 10px;}
    </style>
""", unsafe_allow_html=True)

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
            std_err = math.sqrt(var_sum)
            if survival > 0:
                ci_lower = survival * math.exp(-1.96 * std_err)
                ci_upper = min(1, survival * math.exp(1.96 * std_err))  # Cap at 1
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
        
        n_at_risk -= (n_events + n_censored)
    
    return pd.DataFrame(results)

[... previous plotting functions remain the same ...]

def main():
    st.title('Survival Analysis Calculator')
    
    col1, col2 = st.columns([2, 2])
    
    with col1:
        method = st.radio(
            "Select estimation method:",
            ["Kaplan-Meier", "Nelson-Aalen"]
        )
    
    with col2:
        ci_method = st.radio(
            "Select confidence interval method:",
            ["log", "plain"],
            help="'log' matches R's default method, 'plain' uses linear scale"
        )
    
    st.write("### Data Entry")
    
    n = st.number_input('Enter number of observations:', min_value=1, value=6, max_value=100, key='n_obs', 
                       help="Number of time points to enter")
    
    col1, col2 = st.columns([1, 1])
    
    times = []
    censored = []
    
    for i in range(n):
        with col1:
            times.append(
                st.number_input(f'Time {i+1}:', 
                              min_value=0.0,
                              key=f'time_{i}',
                              step=0.1,
                              format="%.1f")
            )
        with col2:
            censored.append(
                st.checkbox(f'Censored {i+1}?',
                          key=f'cens_{i}')
            )
    
    if st.button('Calculate and Plot'):
        if not times:
            st.error("Please enter at least one time value.")
            return
            
        results_df = calculate_km_estimates(times, censored, ci_method)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = plot_survival_curve(results_df, method)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("### Results Table")
            display_df = format_results_table(results_df)
            st.dataframe(display_df, hide_index=True)
        
        st.write("### Equivalent R Code")
        st.code(generate_r_code(times, censored, method, ci_method), language='r')
        
        st.write("### Equivalent Python Code")
        st.code(generate_python_code(times, censored, method, ci_method), language='python')
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download R Code",
                data=generate_r_code(times, censored, method, ci_method),
                file_name="survival_analysis.R",
                mime="text/plain"
            )
        with col2:
            st.download_button(
                label="Download Python Code",
                data=generate_python_code(times, censored, method, ci_method),
                file_name="survival_analysis.py",
                mime="text/plain"
            )

if __name__ == '__main__':
    main()