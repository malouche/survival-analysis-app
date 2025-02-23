import pandas as pd
import numpy as np
import streamlit as st

def format_number(value):
    """Format numbers to 3 decimal places for floats, no decimals for integers."""
    if isinstance(value, (int, np.integer)):
        return f"{value:d}"
    elif isinstance(value, float):
        return f"{value:.3f}"
    return value

def display_interactive_table(df, height=400):
    """Display an interactive table with formatting."""
    # Format the dataframe
    formatted_df = df.copy()
    for col in formatted_df.columns:
        if formatted_df[col].dtype in ['float64', 'int64']:
            formatted_df[col] = formatted_df[col].apply(format_number)
    
    # Reset index without showing it
    formatted_df = formatted_df.reset_index(drop=True)
    
    # Display interactive table
    st.markdown('<div class="center-table">', unsafe_allow_html=True)
    st.dataframe(
        formatted_df,
        hide_index=True,
        height=height,
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
