import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

st.set_page_config(
    page_title="Survival Analysis Calculator",
    layout="wide"
)

def calculate_km_estimates(times, censored, ci_method='log'):
    sorted_indices = np.argsort(times)
    times = np.array(times)[sorted_indices]
    censored = np.array(censored)[sorted_indices]
    
    unique_times = np.unique(times)
    results = []
    
    n_at_risk = len(times)
    survival = 1.0
    var_sum = 0.0
    
    for t in unique_times:
        mask_time = times == t
        n_events = sum(mask_time & ~censored)
        n_censored = sum(mask_time & censored)
        
        if n_events > 0:
            p = 1 - n_events/n_at_risk
            survival *= p
            var_sum += n_events / (n_at_risk * (n_at_risk - n_events))
        
        if ci_method == 'log':
            std_err = math.sqrt(var_sum)
            if survival > 0:
                ci_lower = survival * math.exp(-1.96 * std_err)
                ci_upper = survival * math.exp(1.96 * std_err)
            else:
                ci_lower = ci_upper = 0
                std_err = float('inf')
        else:
            std_err = survival * math.sqrt(var_sum)
            ci_lower = max(0, survival - 1.96 * std_err)
            ci_upper = min(1, survival + 1.96 * std_err)
            
            if survival == 0:
                std_err = float('inf')
        
        results.append({
            'time': t,
            'n_risk': n_at_risk,
            'n_event': n_events,
            'n_censored': n_censored,
            'survival': survival,
            'std_err': std_err,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })
        
        n_at_risk -= (n_events + n_censored)
    
    return pd.DataFrame(results)

def plot_survival_curve(df, method):
    fig = go.Figure()
    
    # Create step function
    x_steps = [0]
    y_steps = [1]
    ci_x = [0]
    ci_upper = [1]
    ci_lower = [1]
    
    for i in range(len(df)):
        current_time = df['time'].iloc[i]
        current_surv = df['survival'].iloc[i]
        current_upper = df['ci_upper'].iloc[i]
        current_lower = df['ci_lower'].iloc[i]
        
        # Add vertical line
        x_steps.extend([current_time, current_time])
        y_steps.extend([y_steps[-1], current_surv])
        
        # Add horizontal line if not last point
        if i < len(df) - 1:
            next_time = df['time'].iloc[i + 1]
            x_steps.append(next_time)
            y_steps.append(current_surv)
            
        # Add CI points
        ci_x.extend([current_time, current_time])
        ci_upper.extend([ci_upper[-1], current_upper])
        ci_lower.extend([ci_lower[-1], current_lower])
        
        if i < len(df) - 1:
            ci_x.append(next_time)
            ci_upper.append(current_upper)
            ci_lower.append(current_lower)
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=ci_x + ci_x[::-1],
        y=ci_upper + ci_lower[::-1],
        fill='toself',
        fillcolor='rgba(0,0,255,0.1)',
        line=dict(width=0),
        name='95% CI',
        showlegend=True
    ))
    
    # Add survival curve
    fig.add_trace(go.Scatter(
        x=x_steps,
        y=y_steps,
        mode='lines',
        name='Survival Probability',
        line=dict(color='blue', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{method} Survival Estimate',
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            title='Time',
            zeroline=True,
            gridwidth=1
        ),
        yaxis=dict(
            title='Survival Probability',
            range=[0, 1.05],
            zeroline=True,
            gridwidth=1
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def format_results_table(df):
    """Format the results table for display"""
    display_df = df.copy()
    display_df['survival'] = display_df['survival'].map('{:.4f}'.format)
    display_df['std_err'] = display_df['std_err'].apply(lambda x: 'inf' if math.isinf(x) else '{:.4f}'.format(x))
    display_df['95% CI'] = display_df.apply(
        lambda x: '[{:.4f}, {:.4f}]'.format(x['ci_lower'], x['ci_upper']), 
        axis=1
    )
    
    return display_df[[
        'time', 'n_risk', 'n_event', 'n_censored', 
        'survival', 'std_err', '95% CI'
    ]].rename(columns={
        'time': 'Time',
        'n_risk': 'N at Risk',
        'n_event': 'N Events',
        'n_censored': 'N Censored',
        'survival': 'Survival',
        'std_err': 'Std Error'
    })

def main():
    st.title('Survival Analysis Calculator')
    
    method = st.radio(
        "Select estimation method:",
        ["Kaplan-Meier", "Nelson-Aalen"]
    )
    
    ci_method = st.radio(
        "Select confidence interval method:",
        ["log", "plain"],
        help="'log' matches R's default method, 'plain' uses linear scale"
    )
    
    st.write("### Data Entry")
    
    n = st.number_input('Enter number of observations:', min_value=1, value=6)
    
    times = []
    censored = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("##### Enter Times")
        for i in range(n):
            times.append(
                st.number_input(f'Time {i+1}:', 
                              min_value=0.0,
                              key=f'time_{i}',
                              step=0.1)
            )
    
    with col2:
        st.write("##### Censoring Status")
        for i in range(n):
            censored.append(
                st.checkbox(f'Censored {i+1}?',
                          key=f'cens_{i}')
            )
    
    if st.button('Calculate and Plot'):
        if not times:
            st.error("Please enter at least one time value.")
            return
            
        results_df = calculate_km_estimates(times, censored, ci_method)
        
        fig = plot_survival_curve(results_df, method)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### Results Table")
        display_df = format_results_table(results_df)
        st.dataframe(display_df, hide_index=True)

if __name__ == '__main__':
    main()