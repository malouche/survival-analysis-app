import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, NelsonAalenFitter

# Configure the page
st.set_page_config(page_title="Survival Analysis", layout="wide")

def local_css():
    st.markdown("""
        <style>
            .block-container {padding: 2rem;}
            div[data-testid="stExpander"] {
                background-color: #f8f9fa;
                border-left: 5px solid #1f77b4;
                padding: 1rem;
                margin: 1rem 0;
            }
            h1, h2, h3 {color: #2C3E50 !important;}
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

def main():
    local_css()
    
    # Sidebar
    with st.sidebar:
        st.title("Survival Analysis")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Column selection
            time_col = st.selectbox("Select Time Column", options=df.columns)
            status_col = st.selectbox("Select Status Column", options=df.columns)
            
            # Method selection
            method = st.selectbox("Method", ["Kaplan-Meier", "Nelson-Aalen"])
            
            # Alpha level
            alpha = st.slider("Alpha Level", 0.01, 0.20, 0.05)
        
        st.markdown("<div class='footer'>Copyright © 2023 Dhafer Malouche</div>", unsafe_allow_html=True)
    
    # Main panel
    if uploaded_file is not None:
        if method == "Kaplan-Meier":
            kmf = KaplanMeierFitter()
            kmf.fit(df[time_col], df[status_col], alpha=alpha)
            
            # Create plot
            fig = go.Figure()
            
            # Add step function
            times = kmf.timeline
            survival = kmf.survival_function_.values.flatten()
            
            # Create step plot coordinates
            x_step = []
            y_step = []
            for i in range(len(times)):
                if i == 0:
                    x_step.extend([times[i], times[i]])
                    y_step.extend([1.0, survival[i]])
                else:
                    x_step.extend([times[i-1], times[i], times[i]])
                    y_step.extend([survival[i-1], survival[i-1], survival[i]])
            
            fig.add_trace(go.Scatter(
                x=x_step,
                y=y_step,
                mode='lines',
                name='KM Estimate',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Add confidence intervals
            ci_lower = kmf.confidence_interval_.values[:, 0]
            ci_upper = kmf.confidence_interval_.values[:, 1]
            
            # Create step coordinates for CI
            x_ci = []
            y_lower = []
            y_upper = []
            for i in range(len(times)):
                if i == 0:
                    x_ci.extend([times[i], times[i]])
                    y_lower.extend([1.0, ci_lower[i]])
                    y_upper.extend([1.0, ci_upper[i]])
                else:
                    x_ci.extend([times[i-1], times[i], times[i]])
                    y_lower.extend([ci_lower[i-1], ci_lower[i-1], ci_lower[i]])
                    y_upper.extend([ci_upper[i-1], ci_upper[i-1], ci_upper[i]])
            
            fig.add_trace(go.Scatter(
                x=x_ci,
                y=y_lower,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=x_ci,
                y=y_upper,
                mode='lines',
                fill='tonexty',
                line=dict(width=0),
                name=f'{int((1-alpha)*100)}% CI'
            ))
            
            # Add censored points
            censored_times = df[df[status_col] == 0][time_col]
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
            
            fig.update_layout(
                title='Kaplan-Meier Estimate',
                xaxis_title='Time',
                yaxis_title='Survival Probability',
                template='plotly_white',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("Results")
            results = pd.DataFrame({
                'time': kmf.timeline,
                'survival': kmf.survival_function_.values.flatten(),
                'ci_lower': kmf.confidence_interval_.values[:, 0],
                'ci_upper': kmf.confidence_interval_.values[:, 1]
            })
            st.dataframe(results)
            
        else:  # Nelson-Aalen
            naf = NelsonAalenFitter()
            naf.fit(df[time_col], df[status_col], alpha=alpha)
            
            fig = go.Figure()
            
            # Add step function
            times = naf.timeline
            hazard = naf.cumulative_hazard_.values.flatten()
            
            # Create step plot coordinates
            x_step = []
            y_step = []
            for i in range(len(times)):
                if i == 0:
                    x_step.extend([times[i], times[i]])
                    y_step.extend([0.0, hazard[i]])
                else:
                    x_step.extend([times[i-1], times[i], times[i]])
                    y_step.extend([hazard[i-1], hazard[i-1], hazard[i]])
            
            fig.add_trace(go.Scatter(
                x=x_step,
                y=y_step,
                mode='lines',
                name='NA Estimate',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Add confidence intervals
            ci_lower = naf.confidence_interval_.values[:, 0]
            ci_upper = naf.confidence_interval_.values[:, 1]
            
            # Create step coordinates for CI
            x_ci = []
            y_lower = []
            y_upper = []
            for i in range(len(times)):
                if i == 0:
                    x_ci.extend([times[i], times[i]])
                    y_lower.extend([0.0, ci_lower[i]])
                    y_upper.extend([0.0, ci_upper[i]])
                else:
                    x_ci.extend([times[i-1], times[i], times[i]])
                    y_lower.extend([ci_lower[i-1], ci_lower[i-1], ci_lower[i]])
                    y_upper.extend([ci_upper[i-1], ci_upper[i-1], ci_upper[i]])
            
            fig.add_trace(go.Scatter(
                x=x_ci,
                y=y_lower,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=x_ci,
                y=y_upper,
                mode='lines',
                fill='tonexty',
                line=dict(width=0),
                name=f'{int((1-alpha)*100)}% CI'
            ))
            
            fig.update_layout(
                title='Nelson-Aalen Estimate',
                xaxis_title='Time',
                yaxis_title='Cumulative Hazard',
                template='plotly_white',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("Results")
            results = pd.DataFrame({
                'time': naf.timeline,
                'hazard': naf.cumulative_hazard_.values.flatten(),
                'ci_lower': naf.confidence_interval_.values[:, 0],
                'ci_upper': naf.confidence_interval_.values[:, 1]
            })
            st.dataframe(results)

    # About tab
    st.markdown("---")
    st.markdown("""
        **About**  
        **Name:** Dhafer Malouche  
        **Department:** Mathematics and Statistics  
        **Institution:** College of Arts and Sciences, Qatar University  
        **Email:** dhafer.malouche@qu.edu.qa  
        **Website:** [dhafermalouche.net](http://dhafermalouche.net)
    """)

if __name__ == '__main__':
    main()