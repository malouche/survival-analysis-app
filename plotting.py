import plotly.graph_objects as go
import bokeh.plotting as bk
from bokeh.models import ColumnDataSource
import plotly.io as pio

def plot_survival_plotly(df, show_ci=True):
    fig = go.Figure()
    
    # Create step function coordinates
    x_steps = [0]
    y_steps = [1]
    ci_x = [0]
    ci_upper = [1]
    ci_lower = [1]
    
    for i in range(len(df)):
        t = df['time'].iloc[i]
        s = df['survival'].iloc[i]
        
        # Add vertical line
        x_steps.extend([t, t])
        y_steps.extend([y_steps[-1], s])
        
        if show_ci:
            ci_x.extend([t, t])
            ci_upper.extend([ci_upper[-1], df['ci_upper'].iloc[i]])
            ci_lower.extend([ci_lower[-1], df['ci_lower'].iloc[i]])
        
        # Add horizontal line if not last point
        if i < len(df) - 1:
            x_steps.append(df['time'].iloc[i+1])
            y_steps.append(s)
            
            if show_ci:
                ci_x.append(df['time'].iloc[i+1])
                ci_upper.append(df['ci_upper'].iloc[i])
                ci_lower.append(df['ci_lower'].iloc[i])
    
    # Add confidence intervals
    if show_ci:
        fig.add_trace(go.Scatter(
            x=ci_x + ci_x[::-1],
            y=ci_upper + ci_lower[::-1],
            fill='toself',
            fillcolor='rgba(0,0,255,0.1)',
            line=dict(width=0),
            name='95% CI'
        ))
    
    # Add survival curve
    fig.add_trace(go.Scatter(
        x=x_steps,
        y=y_steps,
        mode='lines',
        name='Survival',
        line=dict(color='blue', width=2)
    ))
    
    # Add censored points
    censored_mask = df['n_censored'] > 0
    if any(censored_mask):
        fig.add_trace(go.Scatter(
            x=df.loc[censored_mask, 'time'],
            y=df.loc[censored_mask, 'survival'],
            mode='markers',
            name='Censored',
            marker=dict(symbol='x', size=8, color='black')
        ))
    
    fig.update_layout(
        title='Survival Estimate',
        xaxis_title='Time',
        yaxis_title='Survival Probability',
        yaxis=dict(range=[0, 1.05]),
        template='plotly_white'
    )
    
    return fig

def plot_survival_bokeh(df, show_ci=True):
    p = bk.figure(title='Survival Estimate',
                 x_axis_label='Time',
                 y_axis_label='Survival Probability',
                 y_range=(0, 1.05))
    
    # Create step function coordinates (similar to plotly)
    x_steps = [0]
    y_steps = [1]
    
    for i in range(len(df)):
        t = df['time'].iloc[i]
        s = df['survival'].iloc[i]
        x_steps.extend([t, t])
        y_steps.extend([y_steps[-1], s])
        
        if i < len(df) - 1:
            x_steps.append(df['time'].iloc[i+1])
            y_steps.append(s)
    
    source = ColumnDataSource(data=dict(x=x_steps, y=y_steps))
    
    # Add main line
    p.line('x', 'y', line_color='blue', line_width=2, source=source)
    
    # Add confidence intervals if requested
    if show_ci:
        ci_x = []
        ci_upper = []
        ci_lower = []
        
        for i in range(len(df)):
            t = df['time'].iloc[i]
            ci_x.extend([t, t])
            ci_upper.extend([df['ci_upper'].iloc[i], df['ci_upper'].iloc[i]])
            ci_lower.extend([df['ci_lower'].iloc[i], df['ci_lower'].iloc[i]])
            
            if i < len(df) - 1:
                ci_x.append(df['time'].iloc[i+1])
                ci_upper.append(df['ci_upper'].iloc[i])
                ci_lower.append(df['ci_lower'].iloc[i])
        
        p.patch(ci_x + ci_x[::-1],
               ci_upper + ci_lower[::-1],
               alpha=0.1,
               color='blue')
    
    # Add censored points
    censored_mask = df['n_censored'] > 0
    if any(censored_mask):
        p.scatter(df.loc[censored_mask, 'time'],
                 df.loc[censored_mask, 'survival'],
                 marker='x',
                 size=10,
                 color='black',
                 legend_label='Censored')
    
    return p

def plot_survival_ggplot(df, show_ci=True):
    # Use plotly with a ggplot2 theme
    fig = plot_survival_plotly(df, show_ci)
    fig.update_layout(template='ggplot2')
    return fig