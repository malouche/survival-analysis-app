def generate_python_code(df, methods, alpha, ci_method):
    """Generate Python code for the selected estimation methods."""
    # Convert the DataFrame columns to lists for code generation
    times = df['time'].tolist()
    events = df['event'].tolist()
    
    code = f'''import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from scipy.stats import norm

# Create vectors of time and events
time = {times}
event = {events}

# Create DataFrame
df = pd.DataFrame({{'time': time, 'event': event}})

'''
    
    if 'KM' in methods:
        code += f'''# Kaplan-Meier estimation
kmf = KaplanMeierFitter(alpha={alpha})
kmf.fit(durations=df['time'], event_observed=df['event'])
print("\\nKaplan-Meier Estimation:")
print(kmf.survival_function_)
'''
    
    if 'NA' in methods:
        code += f'''# Nelson-Aalen estimation
naf = NelsonAalenFitter(alpha={alpha})
naf.fit(durations=df['time'], event_observed=df['event'])
print("\\nNelson-Aalen Estimation:")
print(np.exp(-naf.cumulative_hazard_))
'''
    
    return code

def generate_r_code(df, methods, alpha, ci_method):
    """Generate R code for the selected estimation methods."""
    # Convert the DataFrame columns to lists for code generation
    times = df['time'].tolist()
    events = df['event'].tolist()
    
    code = f'''# Create vectors of time and events
time <- c{tuple(times)}
event <- c{tuple(events)}

# Create data frame
data <- data.frame(time = time, event = event)

library(survival)
'''
    
    if 'KM' in methods:
        code += f'''# Kaplan-Meier estimation
km_fit <- survfit(Surv(time, event) ~ 1, data=data, conf.type="{ci_method}",
                  conf.int={1-alpha})
print("Kaplan-Meier Estimation:")
print(km_fit)
'''
    
    if 'NA' in methods:
        code += f'''# Nelson-Aalen estimation
na_fit <- survfit(Surv(time, event) ~ 1, data=data, type="fh",
                  conf.type="{ci_method}", conf.int={1-alpha})
print("\\nNelson-Aalen Estimation:")
print(exp(-na_fit$cumhaz))
'''
    
    return code
