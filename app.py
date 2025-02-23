import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from data import load_data
from estimation import estimate_km, estimate_na
from plotting import plot_with_plotly, plot_with_matplotlib

# Custom CSS injection for tab decoration and centering the data table
st.markdown(
    """
    <style>
    /* Style the tab buttons with borders and a notebook-like appearance */
    div[role="tablist"] > button {
        border: 1px solid #ddd;
        border-radius: 4px;
        margin-right: 5px;
        padding: 5px 10px;
        background-color: #f9f9f9;
    }
    div[role="tablist"] > button:hover {
        background-color: #e0e0e0;
    }
    /* Center the input data table */
    .centered-table {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title('Survival Function Estimator')

with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV (columns: time, event)")
    method = st.selectbox("Select estimation method", ["Kaplan-Meier", "Nelson-Aalen"])
    alpha = st.slider("Select confidence level (alpha)", 0.0, 0.5, 0.05)
    plot_lib = st.selectbox("Select plotting library", ["plotly", "ggplot"])
    if method == "Kaplan-Meier":
        ci_method = st.selectbox("Select CI estimation method", ["Plain", "Arcsine", "Delta"])

if st.sidebar.button("Calculate"):
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if method == "Kaplan-Meier":
            table_df, fitter = estimate_km(df, alpha, ci_method)
            times = fitter.survival_function_.index.values
            survival = fitter.survival_function_['KM_estimate'].values
            ci_lower = fitter.confidence_interval_['KM_estimate_lower_{}'.format(1 - alpha)].values
            ci_upper = fitter.confidence_interval_['KM_estimate_upper_{}'.format(1 - alpha)].values
        else:
            table_df, fitter = estimate_na(df, alpha)
            times = fitter.cumulative_hazard_.index.values
            cum_hazard = fitter.cumulative_hazard_['NA_estimate'].values
            survival = np.exp(-cum_hazard)
            ci_lower = np.exp(-fitter.confidence_interval_['NA_estimate_upper_{}'.format(1 - alpha)].values)
            ci_upper = np.exp(-fitter.confidence_interval_['NA_estimate_lower_{}'.format(1 - alpha)].values)

        python_code = '''
import pandas as pd
from lifelines import KaplanMeierFitter, NelsonAalenFitter

df = pd.read_csv("your_data.csv")

kmf = KaplanMeierFitter(alpha=0.05)
kmf.fit(durations=df['time'], event_observed=df['event'])
print(kmf.survival_function_)

naf = NelsonAalenFitter(alpha=0.05)
naf.fit(durations=df['time'], event_observed=df['event'])
print(np.exp(-naf.cumulative_hazard_))
'''
        r_code = '''
library(survival)
data <- read.csv("your_data.csv")
km_fit <- survfit(Surv(time, event) ~ 1, data=data, conf.int=0.95)
print(km_fit)
na_fit <- survfit(coxph(Surv(time, event) ~ 1, data=data))
surv_est <- exp(-na_fit$cumhaz)
print(surv_est)
'''

        # Create tabs in the main panel
        tabs = st.tabs(["Plot", "Estimation", "Input Data", "R code", "Python code", "About Me", "Feedback"])

        with tabs[0]:
            st.header("Survival Function Plot")
            if plot_lib == "plotly":
                st.plotly_chart(plot_with_plotly(times, survival, ci_lower, ci_upper))
            else:
                st.pyplot(plot_with_matplotlib(times, survival, ci_lower, ci_upper))

        with tabs[1]:
            st.header("Survival Function Estimation")
            st.table(table_df)

        with tabs[2]:
            st.header("Input Data")
            st.markdown("<div class='centered-table'>", unsafe_allow_html=True)
            st.table(df)
            st.markdown("</div>", unsafe_allow_html=True)

        with tabs[3]:
            st.header("R Code")
            st.code(r_code, language='r')

        with tabs[4]:
            st.header("Python Code")
            st.code(python_code, language='python')

        with tabs[5]:
            st.header("About Me")
            st.write("** Dhafer Malouche, Professor of Statistics**")
            st.write("**Department of  Mathematics and Statistics**")
            st.write("**College of Arts and Sciences, Qatar University**")
            st.write("**Email:** [dhafer.malouche@qu.edu.qa](mailto:dhafer.malouche@qu.edu.qa)")
            st.markdown("**Website:** [dhafermalouche.net](http://dhafermalouche.net)")

        with tabs[6]:
            st.header("Feedback")
            with st.form("feedback_form"):
                feedback = st.text_area("Your feedback", height=150)
                submit = st.form_submit_button("Submit")
                if submit:
                    st.write("Thank you for your feedback!")
    else:
        st.error("Please upload a CSV file.")
