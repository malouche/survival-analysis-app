# Install and load necessary packages
if (!require("survival")) install.packages("survival", dependencies = TRUE)

library(survival)

# Load dataset
data(lung)
lung$event <- ifelse(lung$status == 2, 1, 0)

# Create Surv object
surv_obj <- Surv(time = lung$time, event = lung$event)

# -------------------------------
# 1️⃣ Kaplan-Meier Estimator
# -------------------------------

# Plain (Greenwood's Formula)
km_plain <- survfit(surv_obj ~ 1, data = lung, conf.type = "plain")

# Arcsin (Log-Log Transformation)
km_arcsin <- survfit(surv_obj ~ 1, data = lung, conf.type = "arcsin")

# Log-Log Transformation
km_loglog <- survfit(surv_obj ~ 1, data = lung, conf.type = "log-log")

# Extract Kaplan-Meier Results
km_plain_df <- data.frame(
  time = km_plain$time,
  survival = km_plain$surv,
  lower_CI = km_plain$lower,
  upper_CI = km_plain$upper,
  method = "KM - Plain"
)

km_arcsin_df <- data.frame(
  time = km_arcsin$time,
  survival = km_arcsin$surv,
  lower_CI = km_arcsin$lower,
  upper_CI = km_arcsin$upper,
  method = "KM - Arcsin"
)

km_loglog_df <- data.frame(
  time = km_loglog$time,
  survival = km_loglog$surv,
  lower_CI = km_loglog$lower,
  upper_CI = km_loglog$upper,
  method = "KM - Log-Log"
)

# -------------------------------
# 2️⃣ Nelson-Aalen Estimator
# -------------------------------

# Plain (Greenwood's Formula)
na_plain <- survfit(surv_obj ~ 1, data = lung, type = "fh", conf.type = "plain")

# Arcsin (Log-Log Transformation)
na_arcsin <- survfit(surv_obj ~ 1, data = lung, type = "fh", conf.type = "arcsin")

# Log-Log Transformation
na_loglog <- survfit(surv_obj ~ 1, data = lung, type = "fh", conf.type = "log-log")

# Derive Survival Function from Cumulative Hazard: S(t) = exp(-H(t))
na_plain_df <- data.frame(
  time = na_plain$time,
  survival = exp(-na_plain$cumhaz),
  lower_CI = exp(-na_plain$upper),
  upper_CI = exp(-na_plain$lower),
  method = "NA - Plain"
)

na_arcsin_df <- data.frame(
  time = na_arcsin$time,
  survival = exp(-na_arcsin$cumhaz),
  lower_CI = exp(-na_arcsin$upper),
  upper_CI = exp(-na_arcsin$lower),
  method = "NA - Arcsin"
)

na_loglog_df <- data.frame(
  time = na_loglog$time,
  survival = exp(-na_loglog$cumhaz),
  lower_CI = exp(-na_loglog$upper),
  upper_CI = exp(-na_loglog$lower),
  method = "NA - Log-Log"
)

# -------------------------------
# 3️⃣ Combine All Results
# -------------------------------

# Combine KM and NA dataframes
all_results_df <- rbind(
  km_plain_df, km_arcsin_df, km_loglog_df,
  na_plain_df, na_arcsin_df, na_loglog_df
)

# View first few rows
head(all_results_df)

# -------------------------------
# 4️⃣ Save to CSV (Optional)
# -------------------------------

# write.csv(all_results_df, "Survival_Estimates_KM_NA.csv", row.names = FALSE)

# Print the complete data frame
print(all_results_df)

