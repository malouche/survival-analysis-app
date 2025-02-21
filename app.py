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
        mask_time = times == t
        n_events = sum(mask_time & ~censored)
        n_censored = sum(mask_time & censored)
        
        if n_events > 0:
            p = 1 - n_events/n_at_risk
            survival *= p
            var_sum += n_events / (n_at_risk * (n_at_risk - n_events))
        
        # Calculate standard error based on method
        if ci_method == 'log':
            # Log-transformed CI (default in R)
            std_err = sqrt(var_sum)
            if survival > 0:
                ci_lower = survival * exp(-1.96 * std_err)
                ci_upper = survival * exp(1.96 * std_err)
            else:
                ci_lower = ci_upper = 0
        else:
            # Plain CI
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

def calculate_na_estimates(times, censored, ci_method='log'):
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
        mask_time = times == t
        n_events = sum(mask_time & ~censored)
        n_censored = sum(mask_time & censored)
        
        if n_events > 0:
            hazard_increment = n_events/n_at_risk
            cumulative_hazard += hazard_increment
            var_sum += n_events / (n_at_risk * n_at_risk)
        
        survival = exp(-cumulative_hazard)
        
        # Calculate standard error based on method
        if ci_method == 'log':
            # Log-transformed CI
            std_err = sqrt(var_sum)
            if survival > 0:
                ci_lower = survival * exp(-1.96 * std_err)
                ci_upper = survival * exp(1.96 * std_err)
            else:
                ci_lower = ci_upper = 0
        else:
            # Plain CI
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
    """Create step function plot for survival curve"""
    fig = go.Figure()
    
    # Create step function
    x = [0]  # Start at 0
    y = [1]  # Start at 1
    y_upper = [1]
    y_lower = [1]
    
    for i in range(len(df)):
        current_time = df['time'].iloc[i]
        current_surv = df['survival'].iloc[i]
        current_lower = df['ci_lower'].iloc[i]
        current_upper = df['ci_upper'].iloc[i]
        
        # Add vertical line (if not first point)
        if i > 0:
            x.extend([current_time, current_time])
            y.extend([y[-1], current_surv])
            y_lower.extend([y_lower[-1], current_lower])
            y_upper.extend([y_upper[-1], current_upper])
        else:
            x.append(current_time)
            y.append(current_surv)
            y_lower.append(current_lower)
            y_upper.append(current_upper)
        
        # Add horizontal line if not last point
        if i < len(df) - 1:
            next_time = df['time'].iloc[i+1]
            x.append(next_time)
            y.append(current_surv)
            y_lower.append(current_lower)
            y_upper.append(current_upper)
    
    # Add survival curve
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Survival Probability',
        line=dict(color='rgb(31, 119, 180)', width=2)
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=y_upper + y_lower[::-1],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='95% CI',
        showlegend=True
    ))
    
    # Add censored points
    censored_mask = df['n_censored'] > 0
    if any(censored_mask):
        fig.add_trace(go.Scatter(
            x=df.loc[censored_mask, 'time'],
            y=df.loc[censored_mask, 'survival'],
            mode='markers',
            name='Censored',
            marker=dict(
                symbol='plus',
                size=10,
                color='black',
                line=dict(width=2)
            )
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{method} Survival Estimate',
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            title='Time',
            zeroline=True,
            gridwidth=1,
            title_standoff=15
        ),
        yaxis=dict(
            title='Survival Probability',
            range=[0, 1.05],
            zeroline=True,
            gridwidth=1,
            title_standoff=15
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(t=50, l=50, r=30, b=50)
    )
    
    return fig

def format_results_table(df):
    """Format the results table for display"""
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

def generate_r_code(times, censored, method, ci_method='log'):
    """Generate equivalent R code for the analysis"""
    # Create data frame components as strings
    times_str = ", ".join(map(str, times))
    censored_str = ", ".join(map(str, [int(not x) for x in censored]))
    
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

# Create data
times = np.array([{times_str}])
censored = np.array([{censored_str}])

# Initialize and fit the model
{"kmf = KaplanMeierFitter()" if method == "Kaplan-Meier" else "naf = NelsonAalenFitter()"}
{"kmf" if method == "Kaplan-Meier" else "naf"}.fit(times, 
         event_observed=(~censored.astype(bool)),
         conf_type="{ci_method}",
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
            
        # Calculate estimates based on selected method
        if method == "Kaplan-Meier":
            results_df = calculate_km_estimates(times, censored, ci_method)
        else:
            results_df = calculate_na_estimates(times, censored, ci_method)
        
        # Create survival plot
        fig = plot_survival_curve(results_df, method)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display results table
        st.write("### Results Table")
        st.write(f"{method} Survival Estimates")
        display_df = format_results_table(results_df)
        st.dataframe(display_df, hide_index=True)
        
        # Display equivalent R code
        st.write("### Equivalent R Code")
        st.code(generate_r_code(times, censored, method, ci_method), language='r')
        
        # Display equivalent Python code
        st.write("### Equivalent Python Code")
        st.code(generate_python_code(times, censored, method, ci_method), language='python')

if __name__ == '__main__':
    main()