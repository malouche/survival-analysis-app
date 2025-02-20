import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import exp

st.set_page_config(
    page_title="Survival Analysis Calculator",
    layout="wide"
)

def calculate_km_survival(times, censored):
    """Calculate Kaplan-Meier survival estimates"""
    # Sort data
    sorted_indices = np.argsort(times)
    times = np.array(times)[sorted_indices]
    censored = np.array(censored)[sorted_indices]
    
    unique_times = np.unique(times)
    n_at_risk = len(times)
    survival = 1.0
    survival_times = []
    survival_probs = []
    
    for t in unique_times:
        events = sum((times == t) & ~censored)
        if events > 0:
            survival *= (1 - events/n_at_risk)
        n_at_risk -= sum(times == t)
        survival_times.append(t)
        survival_probs.append(survival)
    
    return survival_times, survival_probs

def calculate_na_survival(times, censored):
    """Calculate Nelson-Aalen survival estimates"""
    # Sort data
    sorted_indices = np.argsort(times)
    times = np.array(times)[sorted_indices]
    censored = np.array(censored)[sorted_indices]
    
    unique_times = np.unique(times)
    n_at_risk = len(times)
    cumulative_hazard = 0.0
    survival_times = []
    survival_probs = []
    
    for t in unique_times:
        events = sum((times == t) & ~censored)
        if events > 0:
            cumulative_hazard += events/n_at_risk
        n_at_risk -= sum(times == t)
        survival_times.append(t)
        survival_probs.append(exp(-cumulative_hazard))
    
    return survival_times, survival_probs

def plot_survival_curve(times, probabilities, method):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times,
        y=probabilities,
        mode='lines+markers',
        name='Survival Probability',
        line=dict(color='blue', width=2)
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

def generate_latex_formula(times, probabilities, method):
    latex = r"S(t) = \begin{cases}"
    
    for i, (t, p) in enumerate(zip(times, probabilities)):
        if i == 0:
            latex += f"1, & t < {t:.2f} \\\\"
        latex += f"{p:.3f}, & {t:.2f} \leq t < "
        if i < len(times) - 1:
            latex += f"{times[i+1]:.2f} \\\\"
        else:
            latex += r"\infty \\"
    
    latex += r"\end{cases}"
    
    return latex

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
    
    for i in range(n):
        col1, col2 = st.columns(2)
        with col1:
            time = st.number_input(f'Time {i+1}:', min_value=0.0, key=f'time_{i}')
            times.append(time)
        with col2:
            is_censored = st.checkbox(f'Censored {i+1}?', key=f'cens_{i}')
            censored.append(is_censored)
    
    if st.button('Calculate and Plot'):
        if method == "Kaplan-Meier":
            survival_times, survival_probs = calculate_km_survival(times, censored)
        else:
            survival_times, survival_probs = calculate_na_survival(times, censored)
        
        # Create survival plot
        fig = plot_survival_curve(survival_times, survival_probs, method)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display LaTeX formula
        st.write("### Survival Function Formula")
        latex_formula = generate_latex_formula(survival_times, survival_probs, method)
        st.latex(latex_formula)
        
        # Display data table
        st.write("### Results Table")
        results_df = pd.DataFrame({
            'Time': survival_times,
            'Survival Probability': survival_probs
        })
        st.dataframe(results_df)

if __name__ == '__main__':
    main()
