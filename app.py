import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import exp, sqrt, log

[... previous functions remain the same ...]

def generate_r_code(times, censored, method, ci_method='log'):
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
               conf.type = "{ci_method}",
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

def generate_python_code(times, censored, method, ci_method='log'):
    """Generate equivalent Python code for the analysis"""
    # Create data lists as strings
    times_str = ", ".join(map(str, times))
    censored_str = ", ".join(map(str, [int(x) for x in censored]))
    
    python_code = f"""import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter{"" if method == "Kaplan-Meier" else ", NelsonAalenFitter"}
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn')
sns.set_palette("husl")

# Create data
times = np.array([{times_str}])
censored = np.array([{censored_str}])

# Initialize and fit the model
{"kmf = KaplanMeierFitter()" if method == "Kaplan-Meier" else "naf = NelsonAalenFitter()"}
{"kmf" if method == "Kaplan-Meier" else "naf"}.fit(
    durations=times,
    event_observed=(~censored.astype(bool)),
    label='{method} Estimate'
)

# Print summary
print({"kmf" if method == "Kaplan-Meier" else "naf"}.print_summary())

# Create plot
plt.figure(figsize=(10, 6))
{"kmf" if method == "Kaplan-Meier" else "naf"}.plot(
    ci_show=True,
    ci_alpha=0.2,
    grid=True
)

plt.title('{method} Survival Estimate')
plt.xlabel('Time')
plt.ylabel('Survival Probability')

# Add risk table
{"kmf" if method == "Kaplan-Meier" else "naf"}.plot_survival_table(at_risk_counts=True)

plt.tight_layout()
plt.show()
"""
    return python_code

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
        
        # Display equivalent R code
        st.write("### Equivalent R Code")
        st.code(generate_r_code(times, censored, method, ci_method), language='r')
        
        # Display equivalent Python code
        st.write("### Equivalent Python Code")
        st.code(generate_python_code(times, censored, method, ci_method), language='python')
        
        # Add download buttons for code
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