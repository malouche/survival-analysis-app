import matplotlib.pyplot as plt
from plotly import graph_objs as go

def plot_with_plotly(times, survival, ci_lower, ci_upper):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=survival, mode='lines', name='Survival'))
    fig.add_trace(go.Scatter(x=times, y=ci_lower, mode='lines', name='CI Lower', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=times, y=ci_upper, mode='lines', name='CI Upper', line=dict(dash='dash')))
    fig.update_layout(xaxis_title='Time', yaxis_title='Survival Probability')
    return fig

def plot_with_matplotlib(times, survival, ci_lower, ci_upper):
    # Use a black-and-white theme similar to ggplot2's theme_bw
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()
    ax.step(times, survival, where="post", label="Survival")
    ax.step(times, ci_lower, where="post", linestyle='--', label="CI Lower")
    ax.step(times, ci_upper, where="post", linestyle='--', label="CI Upper")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.legend()
    return fig
