def generate_python_code(df, method, alpha):
    # Convert the DataFrame columns to lists for code generation
    times = df['time'].tolist()
    events = df['event'].tolist()
    
    code = f'''import pandas as pd
from lifelines import KaplanMeierFitter, NelsonAalenFitter
import numpy as np

# Create vectors of time and events
time = {times}
event = {events}

# Create DataFrame
df = pd.DataFrame({{'time': time, 'event': event}})

'''
    if method == "Kaplan-Meier":
        code += f'''kmf = KaplanMeierFitter(alpha={alpha})
kmf.fit(durations=df['time'], event_observed=df['event'])
print(kmf.survival_function_)
'''
    else:  # Nelson-Aalen
        code += f'''naf = NelsonAalenFitter(alpha={alpha})
naf.fit(durations=df['time'], event_observed=df['event'])
print(np.exp(-naf.cumulative_hazard_))
'''
    return code

def generate_r_code(df, method, alpha):
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
    if method == "Kaplan-Meier":
        code += f'''km_fit <- survfit(Surv(time, event) ~ 1, data=data, conf.int={1-alpha})
print(km_fit)
'''
    else:  # Nelson-Aalen
        code += f'''na_fit <- survfit(coxph(Surv(time, event) ~ 1, data=data))
surv_est <- exp(-na_fit$cumhaz)
print(surv_est)
'''
    return code
