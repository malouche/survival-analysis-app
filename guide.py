import streamlit as st

def show_guide():
    st.header("App Description")
    st.write("""
    The Survival Function Estimator is a statistical tool designed for survival analysis. It provides both 
    Kaplan-Meier and Nelson-Aalen estimations of survival functions, which are fundamental methods in survival 
    analysis for estimating the probability of survival past given time points.

    This app allows users to:
    - Upload their survival data
    - Choose between Kaplan-Meier and Nelson-Aalen estimators (or both)
    - Calculate confidence intervals using different methods
    - Visualize results with interactive plots
    - Generate ready-to-use R and Python code
    """)

    st.header("How to Use")
    
    st.subheader("1. Data Upload")
    st.write("""
    - Prepare your data in CSV format with two columns:
        - 'time': survival times (positive numbers)
        - 'event': event indicators (1 for event, 0 for censoring)
    - Click the 'Upload CSV' button in the sidebar to load your data
    """)

    st.subheader("2. Method Selection")
    st.write("""
    In the sidebar, choose one or both estimation methods:
    - **Kaplan-Meier (KM)**: The most common non-parametric estimator of the survival function
    - **Nelson-Aalen (NA)**: An alternative estimator based on cumulative hazard
    """)

    st.subheader("3. Confidence Interval Settings")
    st.write("""
    Configure your confidence intervals:
    - **Alpha Level**: Controls the width of confidence intervals (default: 0.05 for 95% CI)
    - **CI Method**:
        - Plain: Standard normal approximation
        - Arcsin: Arc-sine transformation for better small-sample performance
        - Log-Log: Log-log transformation for improved tail behavior
    - Toggle confidence intervals display using the checkbox
    """)

    st.subheader("4. Viewing Results")
    st.write("""
    Results are organized in tabs:
    - **Plots**: 
        - Individual plots for each selected method
        - Combined plot when both methods are chosen
        - Interactive features for zooming and inspection
    - **Results**: Detailed estimation table with:
        - Time points
        - Number at risk
        - Number of events
        - Survival estimates
        - Confidence intervals
    - **Input Data**: View and verify your uploaded data
    - **Code**: Ready-to-use R and Python code for reproducing the analysis
    """)

    st.subheader("5. Additional Features")
    st.write("""
    - Download plots and tables for use in publications
    - Interactive tables with sorting and filtering capabilities
    - Option to provide feedback
    - Detailed documentation about the methods
    """)

    st.header("Technical Notes")
    st.write("""
    - Survival estimates are presented with 3 decimal places
    - Integer values (like counts) are shown without decimals
    - Estimates and confidence intervals are calculated using standard statistical methods
    - The app handles right-censored data
    """)

    st.header("Example Usage")
    st.write("""
    1. Upload a CSV file with survival data
    2. Select Kaplan-Meier estimation method
    3. Set alpha to 0.05 for 95% confidence intervals
    4. Choose the Plain CI method
    5. Click "Calculate" to view results
    6. Explore different tabs to see various aspects of the analysis
    7. Copy generated code for reproduction in R or Python
    """)

def show_warning():
    st.warning("""
    **Important Notes:**
    - Ensure your data is properly formatted (time, event columns)
    - Time values should be positive numbers
    - Event indicators should be binary (0 or 1)
    - Handle missing values before uploading
    """)
