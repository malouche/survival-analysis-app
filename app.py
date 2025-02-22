import streamlit as st
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, NelsonAalenFitter
import plotly.graph_objects as go
import plotnine as gg

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
    
    # Create step function for survival curve
    x_steps = []
    y_steps = []
    
    # Start at time 0 with survival 1
    x_steps.append(0)
    y_steps.append(1.0)
    
    # Add steps for each time point
    for i in range(len(kmf.timeline)):
        # Add vertical line
        x_steps.extend([kmf.timeline[i], kmf.timeline[i]])
        y_steps.extend([y_steps[-1], kmf.survival_function_.values[i][0]])
    
    # Add survival curve as steps
    fig.add_trace(go.Scatter(
        x=x_steps,
        y=y_steps,
        mode='lines',
        name='Survival Estimate',
        line=dict(color='blue', width=2)
    ))
    
    # Add confidence intervals if available
    if kmf.confidence_interval_ is not None:
        # Lower CI
        x_steps_ci = []
        y_steps_ci_lower = []
        y_steps_ci_upper = []
        
        # Start at time 0
        x_steps_ci.append(0)
        y_steps_ci_lower.append(1.0)
        y_steps_ci_upper.append(1.0)
        
        for i in range(len(kmf.timeline)):
            x_steps_ci.extend([kmf.timeline[i], kmf.timeline[i]])
            y_steps_ci_lower.extend([y_steps_ci_lower[-1], kmf.confidence_interval_.values[i][0]])
            y_steps_ci_upper.extend([y_steps_ci_upper[-1], kmf.confidence_interval_.values[i][1]])
        
        fig.add_trace(go.Scatter(
            x=x_steps_ci,
            y=y_steps_ci_lower,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=x_steps_ci,
            y=y_steps_ci_upper,
            mode='lines',
            fill='tonexty',
            name='95% CI',
            line=dict(width=0)
        ))
    
    # Add censored points
    if censored_times is not None and censored_events is not None:
        censored_mask = censored_events == 0
        censored_times = censored_times[censored_mask]
        survival_at_censored = np.array([
            y_steps[max(i for i, x in enumerate(x_steps) if x <= t)]
            for t in censored_times
        ])
        
        fig.add_trace(go.Scatter(
            x=censored_times,
            y=survival_at_censored,
            mode='markers',
            name='Censored',
            marker=dict(
                symbol='x',
                size=10,
                color='red'
            )
        ))
    
    fig.update_layout(
        title='Survival Function Estimate',
        xaxis_title='Time',
        yaxis_title='Survival Probability',
        yaxis_range=[0, 1.05],
        template='plotly_white',
        width=800,
        height=500
    )
    
    return fig

def plot_survival_ggplot(kmf, censored_times=None, censored_events=None):
    # Create DataFrame for the step function
    steps_x = []
    steps_y = []
    steps_x.append(0)
    steps_y.append(1.0)
    
    for i in range(len(kmf.timeline)):
        steps_x.extend([kmf.timeline[i], kmf.timeline[i]])
        steps_y.extend([steps_y[-1], kmf.survival_function_.values[i][0]])
    
    df = pd.DataFrame({
        'time': steps_x,
        'survival': steps_y
    })
    
    if kmf.confidence_interval_ is not None:
        steps_ci_lower = []
        steps_ci_upper = []
        steps_ci_lower.append(1.0)
        steps_ci_upper.append(1.0)
        
        for i in range(len(kmf.timeline)):
            steps_ci_lower.extend([steps_ci_lower[-1], kmf.confidence_interval_.values[i][0]])
            steps_ci_upper.extend([steps_ci_upper[-1], kmf.confidence_interval_.values[i][1]])
        
        df['ci_lower'] = pd.Series(steps_ci_lower, index=df.index)
        df['ci_upper'] = pd.Series(steps_ci_upper, index=df.index)
    
    # Base plot
    plot = (gg.ggplot(df, gg.aes(x='time', y='survival'))
            + gg.geom_step(color='blue')
            + gg.labs(title='Survival Function Estimate',
                     x='Time',
                     y='Survival Probability')
            + gg.theme_minimal()
            + gg.ylim(0, 1.05))
    
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
        survival_at_censored = np.array([
            steps_y[max(i for i, x in enumerate(steps_x) if x <= t)]
            for t in censored_times
        ])
        
        censored_df = pd.DataFrame({
            'time': censored_times,
            'survival': survival_at_censored
        })
        
        plot = plot + gg.geom_point(
            data=censored_df,
            color='red',
            shape='x',
            size=3
        )
    
    return plot

def get_equivalent_r_code(method, ci_method, data, time_col, event_col):
    """Generate equivalent R code for the analysis using the actual data"""
    # Convert data to R data.frame format
    data_str = "data <- data.frame(\n"
    data_str += f"    {time_col} = c({', '.join(map(str, data[time_col]))}),\n"
    data_str += f"    {event_col} = c({', '.join(map(str, data[event_col]))})\n"
    data_str += ")\n"
    
    if method == "Kaplan-Meier":
        r_code = f"""# Equivalent R code for Kaplan-Meier estimation
library(survival)
library(survminer)

# Your data
{data_str}

# Fit the survival curve
fit <- survfit(
    Surv({time_col}, {event_col}) ~ 1,
    data = data,
    conf.type = "{ci_method.lower()}"
)

# Create the plot
ggsurvplot(
    fit,
    data = data,
    conf.int = TRUE,
    risk.table = TRUE,
    ggtheme = theme_minimal(),
    xlab = "Time",
    ylab = "Survival probability",
    censor = TRUE
)"""
    else:  # Nelson-Aalen
        r_code = f"""# Equivalent R code for Nelson-Aalen estimation
library(survival)
library(survminer)

# Your data
{data_str}

# Fit the Nelson-Aalen estimator
fit <- survfit(
    Surv({time_col}, {event_col}) ~ 1,
    data = data,
    type = "fh",  # Fleming-Harrington estimator
    conf.type = "{ci_method.lower()}"
)

# Create the plot
ggsurvplot(
    fit,
    data = data,
    conf.int = TRUE,
    risk.table = TRUE,
    ggtheme = theme_minimal(),
    xlab = "Time",
    ylab = "Cumulative hazard",
    censor = TRUE
)"""
    
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
                ["Plotly", "ggplot"]
            )
    
    # Main content
    tab1, tab2 = st.tabs(["Analysis", "About"])
    
    with tab1:
        if uploaded_file is not None:
            try:
                times = data[time_col].values
                events = data[event_col].values
                
                if method == "Kaplan-Meier":
                    fitter = KaplanMeierFitter()
                else:
                    fitter = NelsonAalenFitter()
                
                # Fit with the selected CI method
                if ci_method == "Bootstrap":
                    fitter.fit(times, events, alpha=alpha, ci_method=ci_method, n_bootstrap_samples=1000)
                else:
                    fitter.fit(times, events, alpha=alpha, ci_method=ci_method)
                
                # Plot
                if plot_type == "Plotly":
                    fig = plot_survival_plotly(fitter, times, events)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = plot_survival_ggplot(fitter, times, events)
                    st.pyplot(gg.ggplot.draw(fig))
                
                # Show table and R code below the plot
                st.subheader("Survival Table")
                st.dataframe(fitter.survival_function_)
                
                st.subheader("Equivalent R Code")
                r_code = get_equivalent_r_code(method, ci_method, data, time_col, event_col)
                st.code(r_code, language='r')
            
            except Exception as e:
                st.error(f"Error in analysis: {str(e)}")
                st.info("Please check your data format and selected columns.")
    
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
            st.success("Thank you for your feedback!")
    
    # Copyright footer
    st.markdown(
        '<div class="copyright">© 2024 Dhafer Malouche</div>',
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()