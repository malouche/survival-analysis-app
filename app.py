import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, NelsonAalenFitter
import io

[previous code until the main() function remains the same]

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
            try:
                # First check if the file is empty
                file_contents = uploaded_file.getvalue()
                if len(file_contents) == 0:
                    st.error("The uploaded file is empty. Please upload a valid CSV file.")
                    return
                
                # Try to read the CSV file
                try:
                    df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
                    if df.empty:
                        st.error("The uploaded CSV file contains no data.")
                        return
                except pd.errors.EmptyDataError:
                    st.error("The uploaded CSV file contains no data.")
                    return
                except Exception as e:
                    st.error(f"Error reading the CSV file: {str(e)}")
                    return
                
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
            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")
                return
        
        # Footer
        st.markdown(
            "<div class='footer'>Copyright © 2023 Dhafer Malouche</div>",
            unsafe_allow_html=True
        )
    
    # Main panel
    tab1, tab2 = st.tabs(["Analysis", "About"])
    
    with tab1:
        if uploaded_file is not None:
            try:
                # Read the CSV file again (needed because of Streamlit's stateless nature)
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
            except Exception as e:
                st.error(f"An error occurred while analyzing the data: {str(e)}")
    
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