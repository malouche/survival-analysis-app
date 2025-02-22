import streamlit as st
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, NelsonAalenFitter
import plotly.graph_objects as go
import plotnine as gg

# Rest of the imports and CSS remain the same...

def plot_survival_plotly(fitter, censored_times=None, censored_events=None, is_na=False):
    fig = go.Figure()
    
    # Get appropriate function based on estimator type
    if is_na:
        estimate = 1 - np.exp(-fitter.cumulative_hazard_)
        timeline = fitter.timeline
        if fitter.confidence_interval_ is not None:
            ci_lower = 1 - np.exp(-fitter.confidence_interval_['na_upper'])
            ci_upper = 1 - np.exp(-fitter.confidence_interval_['na_lower'])
    else:
        estimate = fitter.survival_function_
        timeline = fitter.timeline
        ci_lower = fitter.confidence_interval_[0]
        ci_upper = fitter.confidence_interval_[1]
    
    # Create step function for survival curve
    x_steps = []
    y_steps = []
    
    # Start at time 0 with survival 1
    x_steps.append(0)
    y_steps.append(1.0)
    
    # Add steps for each time point
    for i in range(len(timeline)):
        x_steps.extend([timeline[i], timeline[i]])
        y_steps.extend([y_steps[-1], estimate.values[i][0]])
    
    # Add survival curve as steps
    fig.add_trace(go.Scatter(
        x=x_steps,
        y=y_steps,
        mode='lines',
        name='Survival Estimate',
        line=dict(color='blue', width=2)
    ))
    
    # Add confidence intervals if available
    if fitter.confidence_interval_ is not None:
        x_steps_ci = []
        y_steps_ci_lower = []
        y_steps_ci_upper = []
        
        # Start at time 0
        x_steps_ci.append(0)
        y_steps_ci_lower.append(1.0)
        y_steps_ci_upper.append(1.0)
        
        for i in range(len(timeline)):
            x_steps_ci.extend([timeline[i], timeline[i]])
            y_steps_ci_lower.extend([y_steps_ci_lower[-1], ci_lower.values[i][0]])
            y_steps_ci_upper.extend([y_steps_ci_upper[-1], ci_upper.values[i][0]])
        
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
        if len(censored_times) > 0:
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

def plot_survival_ggplot(fitter, censored_times=None, censored_events=None, is_na=False):
    # Get appropriate function based on estimator type
    if is_na:
        estimate = 1 - np.exp(-fitter.cumulative_hazard_)
        timeline = fitter.timeline
        if fitter.confidence_interval_ is not None:
            ci_lower = 1 - np.exp(-fitter.confidence_interval_['na_upper'])
            ci_upper = 1 - np.exp(-fitter.confidence_interval_['na_lower'])
    else:
        estimate = fitter.survival_function_
        timeline = fitter.timeline
        ci_lower = fitter.confidence_interval_[0]
        ci_upper = fitter.confidence_interval_[1]
    
    # Create step function data
    steps_x = []
    steps_y = []
    steps_x.append(0)
    steps_y.append(1.0)
    
    for i in range(len(timeline)):
        steps_x.extend([timeline[i], timeline[i]])
        steps_y.extend([steps_y[-1], estimate.values[i][0]])
    
    df = pd.DataFrame({
        'time': steps_x,
        'survival': steps_y
    })
    
    if fitter.confidence_interval_ is not None:
        steps_ci_lower = []
        steps_ci_upper = []
        steps_ci_lower.append(1.0)
        steps_ci_upper.append(1.0)
        
        for i in range(len(timeline)):
            steps_ci_lower.extend([steps_ci_lower[-1], ci_lower.values[i][0]])
            steps_ci_upper.extend([steps_ci_upper[-1], ci_upper.values[i][0]])
        
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
    if fitter.confidence_interval_ is not None:
        plot = plot + gg.geom_ribbon(
            gg.aes(ymin='ci_lower', ymax='ci_upper'),
            alpha=0.2
        )
    
    # Add censored points
    if censored_times is not None and censored_events is not None:
        censored_mask = censored_events == 0
        censored_times = censored_times[censored_mask]
        if len(censored_times) > 0:
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

def get_estimate_table(fitter, is_na=False):
    """Get a formatted table with estimates and CIs"""
    if is_na:
        estimate = 1 - np.exp(-fitter.cumulative_hazard_)
        if fitter.confidence_interval_ is not None:
            ci_lower = 1 - np.exp(-fitter.confidence_interval_['na_upper'])
            ci_upper = 1 - np.exp(-fitter.confidence_interval_['na_lower'])
    else:
        estimate = fitter.survival_function_
        if fitter.confidence_interval_ is not None:
            ci_lower = fitter.confidence_interval_[0]
            ci_upper = fitter.confidence_interval_[1]
    
    # Create table with estimates and CIs
    table = pd.DataFrame({
        'Time': fitter.timeline,
        'Survival': estimate.values.flatten(),
        'CI Lower': ci_lower.values.flatten(),
        'CI Upper': ci_upper.values.flatten()
    })
    
    return table

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
                0.01, 0.20, 0.05
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
                
                is_na = method == "Nelson-Aalen"
                if is_na:
                    fitter = NelsonAalenFitter()
                else:
                    fitter = KaplanMeierFitter()
                
                # Fit with the selected alpha
                fitter.fit(times, events, alpha=alpha)
                
                # Plot
                if plot_type == "Plotly":
                    fig = plot_survival_plotly(fitter, times, events, is_na)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = plot_survival_ggplot(fitter, times, events, is_na)
                    st.pyplot(gg.ggplot.draw(fig))
                
                # Show table with estimates and CIs
                st.subheader("Survival Estimates with Confidence Intervals")
                table = get_estimate_table(fitter, is_na)
                st.dataframe(table.style.format({
                    'Time': '{:.2f}',
                    'Survival': '{:.3f}',
                    'CI Lower': '{:.3f}',
                    'CI Upper': '{:.3f}'
                }))
                
                # Generate and show R code
                st.subheader("Equivalent R Code")
                r_code = get_equivalent_r_code(method, data, time_col, event_col)
                st.code(r_code, language='r')
            
            except Exception as e:
                st.error(f"Error in analysis: {str(e)}")
                st.info("Please check your data format and selected columns.")
    
    # About tab and footer remain the same...

if __name__ == '__main__':
    main()