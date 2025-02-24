import matplotlib.pyplot as plt

def create_survival_plot(times, data, method_name, show_ci=True):
    """Create a single survival plot with confidence intervals."""
    # Use a style that's compatible with newer matplotlib versions
    try:
        plt.style.use('seaborn-v0_8-whitegrid')  # For newer matplotlib versions
    except:
        try:
            plt.style.use('seaborn-whitegrid')  # For older matplotlib versions
        except:
            pass  # Use default style if neither is available
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set color based on method
    color = 'blue' if method_name == 'Kaplan-Meier' else 'red'
    
    # Main survival curve
    ax.step(times, data['estimate'], where='post', 
            label=method_name, color=color)
    
    # Confidence intervals if requested
    if show_ci:
        ax.fill_between(times, data['ci_lower'], data['ci_upper'],
                       step='post', alpha=0.2, color=color)
    
    # Customize the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')
    ax.set_title(f'{method_name} Survival Estimate')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig

def create_combined_plot(times, km_data=None, na_data=None):
    """Create a combined plot of KM and NA estimates without CIs."""
    # Use a style that's compatible with newer matplotlib versions
    try:
        plt.style.use('seaborn-v0_8-whitegrid')  # For newer matplotlib versions
    except:
        try:
            plt.style.use('seaborn-whitegrid')  # For older matplotlib versions
        except:
            pass  # Use default style if neither is available
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if km_data is not None:
        ax.step(times, km_data['estimate'], where='post', 
                label='Kaplan-Meier', color='blue')
    
    if na_data is not None:
        ax.step(times, na_data['estimate'], where='post', 
                label='Nelson-Aalen', color='red')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Combined Survival Estimates')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig
