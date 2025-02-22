import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, NelsonAalenFitter

# Configure the page
st.set_page_config(
    page_title="Survival Analysis",
    layout="wide",
    page_icon="📊"
)

def local_css():
    st.markdown("""
        <style>
            .block-container {padding: 2rem;}
            .css-1d391kg {padding: 2rem 1rem;}
            div[data-testid="stExpander"] {
                background-color: #f8f9fa;
                border-left: 5px solid #1f77b4;
                padding: 1rem;
                margin: 1rem 0;
            }
            h1, h2, h3 {color: #2C3E50 !important;}
            div[data-testid="stDataFrame"] > div {
                border: 1px solid #ddd;
                border-radius: 0.5rem;
                padding: 1rem;
            }
            div[data-testid="stButton"] button {
                background-color: #1f77b4;
                color: white;
                width: 100%;
                padding: 0.5rem;
                margin: 1rem 0;
            }
            div[data-testid="stButton"] button:hover {
                background-color: #1967a9;
            }
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #f8f9fa;
                padding: 1rem;
                text-align: center;
                font-size: 0.8rem;
                color: #666;
            }
        </style>
    """, unsafe_allow_html=True)

def validate_data(df, time_col, censored_col):
    """Validate the uploaded dataset"""
    if time_col not in df.columns or censored_col not in df.columns:
        return False, "Selected columns not found in the dataset"
    
    if not pd.to_numeric(df[time_col], errors='coerce').notnull().all():
        return False, f"'{time_col}' column must contain numeric values"
    
    if not df[censored_col].isin([0, 1]).all():
        return False, f"'{censored_col}' column must contain only 0 or 1"
    
    return True, "Data validation successful"

def plot_survival_curve(df, time_col, censored_col, method='km', ci_method='plain', alpha=0.05):
    """Plot survival curve using the selected method"""
    if method == 'km':
        kmf = KaplanMeierFitter()
        kmf.fit(df[time_col], df[censored_col], alpha=alpha)
        
        fig = go.Figure()
        
        # Add survival curve
        fig.add_trace(go.Scatter(
            x=kmf.timeline,
            y=kmf.survival_function_.values.flatten(),
            mode='lines',
            name='Survival Estimate',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=kmf.timeline,
            y=kmf.confidence_interval_.values[:, 0],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name='Lower CI'
        ))
        
        fig.add_trace(go.Scatter(
            x=kmf.timeline,
            y=kmf.confidence_interval_.values[:, 1],
            mode='lines',
            fill='tonexty',
            line=dict(width=0),
            name=f'{int((1-alpha)*100)}% CI'
        ))
        
        # Add censored points
        censored_times = df[df[censored_col] == 0][time_col]
        if len(censored_times) > 0:
            censored_survival = [kmf.survival_function_.loc[kmf.timeline <= t].iloc[-1] 
                               if len(kmf.survival_function_.loc[kmf.timeline <= t]) > 0 
                               else 1.0 
                               for t in censored_times]
            
            fig.add_trace(go.Scatter(
                x=censored_times,
                y=censored_survival,
                mode='markers',
                name='Censored',
                marker=dict(
                    symbol='x',
                    size=8,
                    color='black',
                    line=dict(width=2)
                )
            ))
    else:
        naf = NelsonAalenFitter()
        naf.fit(df[time_col], df[censored_col], alpha=alpha)
        
        fig = go.Figure()
        
        # Add cumulative hazard curve
        fig.add_trace(go.Scatter(
            x=naf.timeline,
            y=naf.cumulative_hazard_.values.flatten(),
            mode='lines',
            name='Cumulative Hazard',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=naf.timeline,
            y=naf.confidence_interval_.values[:, 0],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=naf.timeline,
            y=naf.confidence_interval_.values[:, 1],
            mode='lines',
            fill='tonexty',
            line=dict(width=0),
            name=f'{int((1-alpha)*100)}% CI'
        ))
    
    fig.update_layout(
        title='Survival Analysis',
        xaxis_title=time_col,
        yaxis_title='Survival Probability' if method == 'km' else 'Cumulative Hazard',
        template='plotly_white',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def generate_r_code(time_col, censored_col, method='km', ci_method='plain', alpha=0.05):
    """Generate equivalent R code"""
    r_code = f"""# Load required libraries
library(survival)
library(survminer)

# Read the data
data <- read.csv("your_data.csv")

# Fit the survival model
surv_obj <- Surv(time = data${time_col}, event = data${censored_col})
"""
    
    if method == 'km':
        r_code += f"""
# Fit Kaplan-Meier model
km_fit <- survfit(surv_obj ~ 1, conf.type = "{ci_method}", conf.int = {1-alpha})

# Plot the survival curve
ggsurvplot(km_fit,
           data = data,
           conf.int = TRUE,
           risk.table = TRUE,
           censor = TRUE)
"""
    else:
        r_code += f"""
# Fit Nelson-Aalen model
na_fit <- survfit(surv_obj ~ 1, type = "fleming-harrington", conf.int = {1-alpha})

# Plot the cumulative hazard
ggsurvplot(na_fit,
           data = data,
           conf.int = TRUE,
           risk.table = TRUE,
           censor = TRUE,
           fun = "cumhaz")
"""
    return r_code

def main():
    local_css()
    
    # Sidebar
    with st.sidebar:
        st.title("Survival Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload your dataset in CSV format"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Column selection
            st.subheader("Data Configuration")
            with st.expander("How to Configure Data", expanded=True):
                st.markdown("""
                    1. **Time Column:** Select the column containing time-to-event data
                    2. **Status Column:** Select the column indicating event status (0=censored, 1=event)
                """)
            
            time_col = st.selectbox(
                "Select Time Column",
                options=df.columns,
                help="Column containing time-to-event data"
            )
            
            censored_col = st.selectbox(
                "Select Status Column",
                options=df.columns,
                help="Column containing event status (0=censored, 1=event)"
            )
            
            # Method selection
            method = st.selectbox(
                "Estimation Method",
                ["Kaplan-Meier (KM)", "Nelson-Aalen (NA)"],
                help="Choose the survival estimation method"
            )
            
            # CI method selection (only for KM for now)
            if method == "Kaplan-Meier (KM)":
                ci_method = st.selectbox(
                    "Confidence Interval Method",
                    ["Plain", "Arcsine", "Delta method"],
                    help="Choose the method for calculating confidence intervals"
                )
            
            # Alpha level
            alpha = st.slider(
                "Significance Level (α)",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                help="Set the significance level for confidence intervals"
            )
        
        # Footer
        st.markdown(
            "<div class='footer'>Copyright © 2023 Dhafer Malouche</div>",
            unsafe_allow_html=True
        )
    
    # Main panel
    tab1, tab2 = st.tabs(["Analysis", "About"])
    
    with tab1:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Validate data
            is_valid, message = validate_data(df, time_col, censored_col)
            
            if is_valid:
                # Display the plot
                fig = plot_survival_curve(
                    df,
                    time_col,
                    censored_col,
                    method='km' if 'KM' in method else 'na',
                    ci_method=ci_method.lower() if 'KM' in method else 'plain',
                    alpha=alpha
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display data table
                st.subheader("Analysis Results")
                if 'KM' in method:
                    kmf = KaplanMeierFitter()
                    kmf.fit(df[time_col], df[censored_col], alpha=alpha)
                    results = pd.DataFrame({
                        'time': kmf.timeline,
                        'estimate': kmf.survival_function_.values.flatten(),
                        'std': np.sqrt(kmf.variance_),
                        'ci_lower': kmf.confidence_interval_.values[:, 0],
                        'ci_upper': kmf.confidence_interval_.values[:, 1]
                    })
                else:
                    naf = NelsonAalenFitter()
                    naf.fit(df[time_col], df[censored_col], alpha=alpha)
                    results = pd.DataFrame({
                        'time': naf.timeline,
                        'estimate': naf.cumulative_hazard_.values.flatten(),
                        'std': np.sqrt(naf.variance_),
                        'ci_lower': naf.confidence_interval_.values[:, 0],
                        'ci_upper': naf.confidence_interval_.values[:, 1]
                    })
                st.dataframe(results)
                
                # R code generation
                if st.button("Show R Code"):
                    r_code = generate_r_code(
                        time_col,
                        censored_col,
                        method='km' if 'KM' in method else 'na',
                        ci_method=ci_method.lower() if 'KM' in method else 'plain',
                        alpha=alpha
                    )
                    st.code(r_code, language='r')
            else:
                st.error(message)
    
    with tab2:
        st.header("About")
        
        # Contact information
        st.subheader("Contact Information")
        st.markdown("""
            **Name:** Dhafer Malouche  
            **Title:** Professor of Statistics  
            **Department:** Mathematics and Statistics  
            **Institution:** College of Arts and Sciences, Qatar University  
            **Email:** [dhafer.malouche@qu.edu.qa](mailto:dhafer.malouche@qu.edu.qa)  
            **Website:** [dhafermalouche.net](http://dhafermalouche.net)
        """)
        
        # Feedback form
        st.subheader("Feedback")
        feedback = st.text_area(
            "Please share your comments, suggestions, or questions:",
            height=150
        )
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")

if __name__ == '__main__':
    main()