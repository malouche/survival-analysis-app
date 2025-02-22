import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import bokeh.plotting as bk
import math
import io
import base64
from datetime import datetime

st.set_page_config(page_title="Survival Analysis Calculator", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0 1rem;}
    .stRadio > div {flex-direction: row;}
    .stRadio > div > label {margin-right: 2rem;}
    .css-1d391kg {padding: 1rem;}
    .sidebar .sidebar-content {background-color: #f5f5f5;}
    .contact-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        padding: 10px;
        background-color: #f5f5f5;
        width: 100%;
        text-align: left;
        font-size: 0.8em;
    }
    .table-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def about():
    st.title("About")
    
    st.markdown("""
    <div class="contact-info">
        <h3>Contact Information</h3>
        <p><strong>Dhafer Malouche</strong><br>
        Professor of Statistics<br>
        Department of Mathematics and Statistics<br>
        College of Arts and Sciences<br>
        Qatar University</p>
        
        <p>📧 Email: <a href="mailto:dhafer.malouche@qu.edu.qa">dhafer.malouche@qu.edu.qa</a><br>
        🌐 Website: <a href="http://dhafermalouche.net" target="_blank">dhafermalouche.net</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("### Send me your comments")
    user_name = st.text_input("Name")
    user_email = st.text_input("Email")
    user_message = st.text_area("Message")
    if st.button("Send"):
        st.success("Thank you for your feedback! I will get back to you soon.")

def calculate_survival(df, method='km', ci_method='log', alpha=0.05):
    times = df['time'].values
    events = (~df['censored']).values
    
    sorted_indices = np.argsort(times)
    times = times[sorted_indices]
    events = events[sorted_indices]
    
    unique_times = np.unique(times)
    n_at_risk = len(times)
    survival = 1.0
    var_sum = 0.0
    results = []
    
    for t in unique_times:
        mask = times == t
        n_events = sum(mask & events)
        n_censored = sum(mask & ~events)
        
        if n_events > 0:
            if method == 'km':
                p = 1 - n_events/n_at_risk
                survival *= p
            else:  # Nelson-Aalen
                h = n_events/n_at_risk
                survival = survival * math.exp(-h)
            
            var_sum += n_events / (n_at_risk * (n_at_risk - n_events))
        
        # Calculate CI based on method
        z = -st.norm.ppf(alpha/2)
        if ci_method == 'plain':
            std_err = survival * math.sqrt(var_sum)
            ci_lower = max(0, survival - z * std_err)
            ci_upper = min(1, survival + z * std_err)
        elif ci_method == 'log':
            std_err = math.sqrt(var_sum)
            if survival > 0:
                ci_lower = survival * math.exp(-z * std_err)
                ci_upper = min(1, survival * math.exp(z * std_err))
            else:
                ci_lower = ci_upper = 0
        elif ci_method == 'arcsin':
            phi = math.asin(math.sqrt(survival))
            std_err = math.sqrt(var_sum) / (2 * math.sqrt(survival * (1 - survival)))
            ci_lower = math.sin(max(0, phi - z * std_err))**2
            ci_upper = math.sin(min(math.pi/2, phi + z * std_err))**2
        
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

def plot_survival(df, plot_type='plotly', show_ci=True):
    if plot_type == 'plotly':
        return plot_survival_plotly(df, show_ci)
    elif plot_type == 'bokeh':
        return plot_survival_bokeh(df, show_ci)
    else:  # ggplot theme
        return plot_survival_ggplot(df, show_ci)

[... previous plotting functions ...]

def generate_r_code(df, method, ci_method, alpha):
    code = f"""# Load required libraries
library(survival)
library(survminer)

# Create data frame
data <- data.frame(
    time = c({', '.join(map(str, df['time']))}),
    status = c({', '.join(map(str, [1 if not x else 0 for x in df['censored']]))})
)

# Fit survival curve
fit <- survfit(Surv(time, status) ~ 1, 
               data = data,
               conf.type = "{ci_method}",
               type = {"'fleming-harrington'" if method == 'na' else "'kaplan-meier'"},
               conf.int = {1-alpha})

# Print summary
print(summary(fit))

# Create plot
ggsurvplot(fit,
           data = data,
           conf.int = TRUE,
           risk.table = TRUE,
           ggtheme = theme_minimal())
"""
    return code

def main():
    st.sidebar.title("Survival Analysis Settings")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Method selection
        method = st.sidebar.radio("Select Method", 
                                ["Kaplan-Meier", "Nelson-Aalen"],
                                format_func=lambda x: "Kaplan-Meier" if x == "km" else "Nelson-Aalen")
        
        # CI method selection
        ci_options = ["plain", "log", "arcsin"] if method == "km" else ["plain", "log"]
        ci_method = st.sidebar.selectbox("Confidence Interval Method", 
                                       ci_options,
                                       format_func=lambda x: x.capitalize())
        
        # Alpha slider
        alpha = st.sidebar.slider("Alpha Level", 0.01, 0.20, 0.05)
        
        # Plot type selection
        plot_type = st.sidebar.selectbox("Plot Style", 
                                       ["plotly", "bokeh", "ggplot"],
                                       format_func=str.capitalize)
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.title("Survival Analysis Results")
            results = calculate_survival(df, method, ci_method, alpha)
            fig = plot_survival(results, plot_type)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("### Results Table")
            st.dataframe(format_results_table(results))
            
            if st.button("Show R Code"):
                st.code(generate_r_code(df, method, ci_method, alpha), language='r')
    
    # Footer
    st.markdown("""
        <div class="footer">
            © 2024 Dhafer Malouche
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    page = st.sidebar.selectbox("Navigation", ["Home", "About"])
    
    if page == "Home":
        main()
    else:
        about()