import streamlit as st
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, NelsonAalenFitter
import plotly.graph_objects as go
import bokeh.plotting as bkp
from bokeh.models import ColumnDataSource
import plotnine as gg
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import base64
from io import StringIO

# Configure the page
st.set_page_config(
    page_title="Survival Analysis App",
    layout="wide"
)

# Custom CSS
def local_css():
    st.markdown("""
        <style>
        .block-container {
            padding: 2rem;
        }
        
        .stDataFrame {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        
        .about-section {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 8px;
            margin: 2rem 0;
        }
        
        .contact-form {
            background-color: #fff;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .copyright {
            position: fixed;
            bottom: 0;
            left: 0;
            padding: 1rem;
            background-color: #f8f9fa;
            width: 100%;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

def plot_survival_plotly(kmf, censored_times=None, censored_events=None):
    fig = go.Figure()
    
    # Add survival curve
    fig.add_trace(go.Scatter(
        x=kmf.timeline,
        y=kmf.survival_function_.values.flatten(),
        mode='lines',
        name='Survival Estimate',
        line=dict(color='blue')
    ))
    
    # Add confidence intervals
    if kmf.confidence_interval_ is not None:
        fig.add_trace(go.Scatter(
            x=kmf.timeline,
            y=kmf.confidence_interval_.values[:, 0],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=kmf.timeline,
            y=kmf.confidence_interval_.values[:, 1],
            mode='lines',
            fill='tonexty',
            name='95% CI',
            line=dict(width=0)
        ))
    
    # Add censored points
    if censored_times is not None and censored_events is not None:
        censored_mask = censored_events == 0
        censored_times = censored_times[censored_mask]
        survival_at_censored = np.interp(censored_times, 
                                       kmf.timeline,
                                       kmf.survival_function_.values.flatten())
        
        fig.add_trace(go.Scatter(
            x=censored_times,
            y=survival_at_censored,
            mode='markers',
            name='Censored',
            marker=dict(
                symbol='plus',
                size=10,
                color='red'
            )
        ))
    
    fig.update_layout(
        title='Survival Function Estimate',
        xaxis_title='Time',
        yaxis_title='Survival Probability',
        template='plotly_white'
    )
    
    return fig

def plot_survival_bokeh(kmf, censored_times=None, censored_events=None):
    p = bkp.figure(title='Survival Function Estimate',
                   x_axis_label='Time',
                   y_axis_label='Survival Probability')
    
    # Add survival curve
    source = ColumnDataSource({
        'x': kmf.timeline,
        'y': kmf.survival_function_.values.flatten()
    })
    p.line('x', 'y', line_color='blue', legend_label='Survival Estimate', source=source)
    
    # Add confidence intervals
    if kmf.confidence_interval_ is not None:
        p.patch(
            x=np.concatenate([kmf.timeline, kmf.timeline[::-1]]),
            y=np.concatenate([
                kmf.confidence_interval_.values[:, 0],
                kmf.confidence_interval_.values[::-1, 1]
            ]),
            alpha=0.2,
            color='blue',
            legend_label='95% CI'
        )
    
    # Add censored points
    if censored_times is not None and censored_events is not None:
        censored_mask = censored_events == 0
        censored_times = censored_times[censored_mask]
        survival_at_censored = np.interp(censored_times,
                                       kmf.timeline,
                                       kmf.survival_function_.values.flatten())
        
        p.scatter(censored_times,
                 survival_at_censored,
                 color='red',
                 marker='plus',
                 size=10,
                 legend_label='Censored')
    
    p.legend.location = "top_right"
    return p

def plot_survival_ggplot(kmf, censored_times=None, censored_events=None):
    # Create DataFrame for ggplot
    df = pd.DataFrame({
        'time': kmf.timeline,
        'survival': kmf.survival_function_.values.flatten()
    })
    
    if kmf.confidence_interval_ is not None:
        df['ci_lower'] = kmf.confidence_interval_.values[:, 0]
        df['ci_upper'] = kmf.confidence_interval_.values[:, 1]
    
    # Base plot
    plot = (gg.ggplot(df, gg.aes(x='time', y='survival'))
            + gg.geom_line(color='blue')
            + gg.labs(title='Survival Function Estimate',
                     x='Time',
                     y='Survival Probability')
            + gg.theme_minimal())
    
    # Add confidence intervals
    if kmf.confidence_interval_ is not None:
        plot = plot + gg.geom_ribbon(
            gg.aes(ymin='ci_lower', ymax='ci_upper'),
            alpha=0.2
        )
    
    # Add censored points
    if censored_times is not None and censored_events is not None:
        censored_mask = censored_events == 0
        censored_times = censored_times[censored_mask]
        survival_at_censored = np.interp(censored_times,
                                       kmf.timeline,
                                       kmf.survival_function_.values.flatten())
        
        censored_df = pd.DataFrame({
            'time': censored_times,
            'survival': survival_at_censored
        })
        
        plot = plot + gg.geom_point(
            data=censored_df,
            color='red',
            shape='+'
        )
    
    return plot

def generate_r_code(method, ci_method):
    r_code = """
library(survival)
library(survminer)

# Read the data
data <- read.csv('your_data.csv')

# Fit the survival curve
fit <- survfit(Surv(time, event) ~ 1, 
               data = data,
               conf.type = '{ci_method}')

# Plot
ggsurvplot(fit,
           data = data,
           conf.int = TRUE,
           risk.table = TRUE,
           censored = TRUE)
""".format(ci_method=ci_method.lower())
    
    return r_code

def main():
    local_css()
    
    # Sidebar
    with st.sidebar:
        st.header("Analysis Settings")
        
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="File should contain columns: time, event"
        )
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            
            time_col = st.selectbox(
                "Select time column",
                data.columns
            )
            
            event_col = st.selectbox(
                "Select event column",
                data.columns
            )
            
            method = st.radio(
                "Select estimation method",
                ["Kaplan-Meier", "Nelson-Aalen"]
            )
            
            alpha = st.slider(
                "Confidence level (α)",
                0.01, 0.99, 0.05
            )
            
            if method == "Kaplan-Meier":
                ci_method = st.selectbox(
                    "Confidence interval method",
                    ["Plain", "Arcsine", "Delta", "Bootstrap"]
                )
            else:
                ci_method = st.selectbox(
                    "Confidence interval method",
                    ["Plain", "Delta"]
                )
            
            plot_type = st.radio(
                "Select plotting library",
                ["Plotly", "Bokeh", "ggplot"]
            )
    
    # Main content
    tab1, tab2 = st.tabs(["Analysis", "About"])
    
    with tab1:
        if uploaded_file is not None:
            times = data[time_col].values
            events = data[event_col].values
            
            if method == "Kaplan-Meier":
                fitter = KaplanMeierFitter()
            else:
                fitter = NelsonAalenFitter()
            
            fitter.fit(times, events, alpha=alpha)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if plot_type == "Plotly":
                    fig = plot_survival_plotly(fitter, times, events)
                    st.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Bokeh":
                    fig = plot_survival_bokeh(fitter, times, events)
                    st.bokeh_chart(fig, use_container_width=True)
                else:
                    fig = plot_survival_ggplot(fitter, times, events)
                    st.pyplot(gg.ggplot.draw(fig))
            
            with col2:
                st.subheader("Survival Table")
                st.dataframe(fitter.survival_function_)
                
                if st.button("Show R code"):
                    r_code = generate_r_code(method, ci_method)
                    st.code(r_code, language='r')
    
    with tab2:
        st.header("About")
        
        st.markdown("""
        ### Contact Information
        **Dhafer Malouche**  
        Professor of Statistics  
        Department of Mathematics and Statistics  
        College of Arts and Sciences  
        Qatar University
        
        **Email:** [dhafer.malouche@qu.edu.qa](mailto:dhafer.malouche@qu.edu.qa)  
        **Website:** [dhafermalouche.net](http://dhafermalouche.net)
        """)
        
        st.subheader("Send Comments")
        comment = st.text_area("Your comments")
        email = st.text_input("Your email")
        
        if st.button("Submit"):
            # Here you would implement the logic to send the comment
            st.success("Thank you for your feedback!")
    
    # Copyright footer
    st.markdown(
        '<div class="copyright">© 2024 Dhafer Malouche</div>',
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()