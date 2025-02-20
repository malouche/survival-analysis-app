import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import exp, sqrt

st.set_page_config(
    page_title="Survival Analysis Calculator",
    layout="wide"
)

def generate_r_code(times, censored, method):
    """Generate equivalent R code for the analysis"""
    # Create data vectors as strings
    times_str = ", ".join(map(str, times))
    censored_str = ", ".join(map(str, [int(not x) for x in censored]))  # R uses 1 for event, 0 for censored
    
    r_code = f"""# Load required libraries
library(survival)
library(survminer)

# Create data vectors
times <- c({times_str})
status <- c({censored_str})  # 1 = event, 0 = censored

# Fit survival curve
fit <- survfit(Surv(times, status) ~ 1, 
               type = {"'fleming-harrington'" if method == "Nelson-Aalen" else "'kaplan-meier'"})

# Print summary
print(summary(fit))

# Create plot
ggsurvplot(fit,
           conf.int = TRUE,
           risk.table = TRUE,
           xlab = "Time",
           ylab = "Survival Probability",
           title = "{method} Survival Estimate")
"""
    return r_code

def generate_python_code(times, censored, method):
    """Generate equivalent Python code for the analysis"""
    # Create data lists as strings
    times_str = ", ".join(map(str, times))
    censored_str = ", ".join(map(str, [int(x) for x in censored]))
    
    python_code = f"""import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter{"" if method == "Kaplan-Meier" else ", NelsonAalenFitter"}
import matplotlib.pyplot as plt

# Create data
times = np.array([{times_str}])
censored = np.array([{censored_str}])

# Initialize and fit the model
{"kmf = KaplanMeierFitter()" if method == "Kaplan-Meier" else "naf = NelsonAalenFitter()"}
{"kmf" if method == "Kaplan-Meier" else "naf"}.fit(times, 
         event_observed=(~censored.astype(bool)),
         label='{method} Estimate')

# Print summary
print({"kmf" if method == "Kaplan-Meier" else "naf"}.print_summary())

# Create plot
plt.figure(figsize=(10, 6))
{"kmf" if method == "Kaplan-Meier" else "naf"}.plot(ci_show=True)
plt.title('{method} Survival Estimate')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()
"""
    return python_code

[... rest of the previous functions remain the same ...]

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
        
        # Display equivalent R code
        st.write("### Equivalent R Code")
        st.code(generate_r_code(times, censored, method), language='r')
        
        # Display equivalent Python code
        st.write("### Equivalent Python Code")
        st.code(generate_python_code(times, censored, method), language='python')
        
        # Add download buttons for the code
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download R Code",
                data=generate_r_code(times, censored, method),
                file_name="survival_analysis.R",
                mime="text/plain"
            )
        with col2:
            st.download_button(
                label="Download Python Code",
                data=generate_python_code(times, censored, method),
                file_name="survival_analysis.py",
                mime="text/plain"
            )

if __name__ == '__main__':
    main()