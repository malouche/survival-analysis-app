import streamlit as st
import numpy as np
from data import load_data
from estimation import estimate_km, estimate_na, combine_estimates
from plotting import create_survival_plot, create_combined_plot
from code_generator import generate_python_code, generate_r_code
from table_utils import display_interactive_table
from about import show_about
from feedback import show_feedback
from guide import show_guide, show_warning

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    .center-table {
        margin: 0 auto;
        max-width: 1000px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Survival Function Estimator')

# Sidebar inputs
with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV (columns: time, event)")
    
    methods = st.multiselect(
        "Select estimation method(s)",
        ["Kaplan-Meier", "Nelson-Aalen"],
        default=["Kaplan-Meier"]
    )
    
    method_map = {"Kaplan-Meier": "KM", "Nelson-Aalen": "NA"}
    selected_methods = [method_map[m] for m in methods]
    
    alpha = st.slider("Select confidence level (alpha)", 0.0, 0.5, 0.05)
    ci_method = st.selectbox(
        "Select CI estimation method",
        ["Plain", "Arcsin", "Log-Log"]
    )
    
    show_ci = st.checkbox("Show confidence intervals", value=True)

if st.sidebar.button("Calculate"):
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        km_results = na_results = None
        km_data = na_data = None
        
        if "KM" in selected_methods:
            km_results, kmf = estimate_km(df, alpha, ci_method.lower())
            km_data = {
                'estimate': km_results['KM_estimate'],
                'ci_lower': km_results['KM_CI_lower'],
                'ci_upper': km_results['KM_CI_upper']
            }
        
        if "NA" in selected_methods:
            na_results, naf = estimate_na(df, alpha, ci_method.lower())
            na_data = {
                'estimate': na_results['NA_estimate'],
                'ci_lower': na_results['NA_CI_lower'],
                'ci_upper': na_results['NA_CI_upper']
            }
        
        combined_results = combine_estimates(km_results, na_results)
        
        # Create main tabs
        tabs = st.tabs(["Plots", "Results", "Input Data", "Code", "About", "Feedback"])
        
        with tabs[0]:
            plot_methods = selected_methods.copy()
            if len(plot_methods) > 1:
                plot_methods.append("Combined")
            
            plot_tabs = st.tabs(plot_methods)
            
            for i, method in enumerate(plot_methods):
                with plot_tabs[i]:
                    if method == "KM":
                        fig = create_survival_plot(
                            km_results['time'], km_data, 
                            "Kaplan-Meier", show_ci
                        )
                        st.pyplot(fig)
                    elif method == "NA":
                        fig = create_survival_plot(
                            na_results['time'], na_data,
                            "Nelson-Aalen", show_ci
                        )
                        st.pyplot(fig)
                    else:  # Combined plot
                        fig = create_combined_plot(
                            combined_results['time'], km_data, na_data
                        )
                        st.pyplot(fig)
        
        with tabs[1]:
            st.header("Survival Function Estimation")
            display_interactive_table(combined_results)
        
        with tabs[2]:
            st.header("Input Data")
            display_interactive_table(df, height=300)
        
        with tabs[3]:
            code_tabs = st.tabs(["Python Code", "R Code"])
            with code_tabs[0]:
                st.code(generate_python_code(df, selected_methods, alpha, ci_method.lower()),
                       language='python')
            with code_tabs[1]:
                st.code(generate_r_code(df, selected_methods, alpha, ci_method.lower()),
                       language='r')
        
         
        with tabs[4]:
            show_guide()
            show_warning()
            show_about()
            
        with tabs[5]:
            show_feedback()
            
    else:
        st.error("Please upload a CSV file.")
