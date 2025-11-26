rm(list = ls())
library(lubridate)
library(ggplot2)
library(dplyr)
library(brms)
library(purrr)
library(parallel)
library(posterior)
library(tidyr)
setwd("C:/Users/Käyttäjä/Desktop/BDA/models")
source("C:/Users/Käyttäjä/Desktop/BDA/models/Functions.R")

# Comment this out if cmdstan works
# install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/",getOption("repos")))
library(cmdstanr)
#dir.create(file.path("~/cmdstan"), showWarnings = FALSE)
#install_cmdstan(dir = "~/cmdstan", overwrite = TRUE)
set_cmdstan_path("~/cmdstan/cmdstan-2.37.0")

cores <- max(1, parallel::detectCores() - 1)

#########################
#### Preprocess Data ####
#########################

file_path <- "C:/Users/Käyttäjä/Desktop/BDA/data/processed/Shiller_cleaned.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE) %>%
  mutate(
    Date = as.Date(Date),
    inv_CAPE = 1 / CAPE,
    CAPE_scaled = scale(CAPE),
    Inflation_scaled = scale(Inflation)
  ) %>%
  filter(complete.cases(.)) %>%
  arrange(Date)

train_start <- as.Date("1881-01-01")
test_start  <- as.Date("1990-01-01")
test_end    <- as.Date("2015-09-01")

rmse_list     <- c()
predictions   <- c()
actuals       <- c()
lowers        <- c()
uppers        <- c()
lpds         <- c()
dates         <- as.Date(character())
diagnostics   <- list()

# Generate monthly dates, then take every 6th month as a test point
all_test_dates <- seq(from = test_start, to = test_end, by = "month")
test_dates <- all_test_dates[seq(1, length(all_test_dates), by = 6)]
n_iter <- length(test_dates)

window_size <- 120

######################
#### Define Model ####
######################

# Formula for bayesian model
formula <- bf(
  Real_Return_10Y ~ 1 + CAPE + (1 + CAPE | Inflation_Category),
  family = "gaussian",
  center = FALSE
)

get_prior(formula, data = df)
priors <- 
  c(prior(normal(-0.5, 0.5), class = "b", coef = "CAPE"),
    prior(normal(14,5), class = "b", coef = "Intercept"),
    prior(lkj(1), class = "cor"))

###########################
#### Fit Initial Model ####
###########################

initial_test_date <- as.Date(test_dates[1])
train_end <- initial_test_date %m-% lubridate::period(years = 10, months = 1)
train <- df %>% 
  filter(Date >= train_start, Date <= train_end)

model <- brm(
  formula = formula,
  prior = priors,
  data = train,
  chains = 3,
  iter = 2000,
  warmup = 1000,
  backend = "cmdstanr",
  adapt_delta = 0.99,
  cores = cores,
  max_treedepth = 20,
  seed = 1
)

##################################################
#### Rolling Forecast Loop With Model Updates ####
##################################################

for (i in seq_along(test_dates)) {
  
  start <- proc.time()
  
  current_date <- as.Date(test_dates[i])
  test_sample <- df %>%
    filter(Date == current_date)
  
  # Predict
  preds <- generate_prediction(model, test_sample)
  y_pred <- preds[1]
  lower <- preds[2]
  upper <- preds[3]
  y_true <- test_sample$Real_Return_10Y
  lpd <- compute_log_pred_density(model, test_sample)
  
  # Store results
  predictions   <- c(predictions, as.numeric(y_pred))
  actuals       <- c(actuals, as.numeric(y_true))
  lowers        <- c(lowers, lower)
  uppers        <- c(uppers, upper)
  dates         <- c(dates, current_date)
  lpds          <- c(lpds, lpd)
  
  # Store convergence diagnostics
  diagnostics[[i]] <- convergence_diagnostics(current_date, model)
  
  # Advance training window by 6 months between test points
  train_end <- train_end %m+% lubridate::period(months = 6)
  train <- df %>%
    filter(Date >= train_start, Date <= train_end)
  
  # Update model
  if (i != n_iter){
    model <- update(
      model,
      newdata = train,
      recompile = FALSE
    )
  }
  
  end <- proc.time()
  elapsed <- (end - start)["elapsed"]
  
  cat("\nITERATION STEP", i, "/", n_iter, "COMPLETED IN", elapsed, "SECONDS\n")
}

# Convert results to data frame
results_df <- data.frame(
  Date      = dates,
  Predicted = predictions,
  Actual    = actuals,
  Upper     = uppers,
  Lower     = lowers,
  Lpds      = lpds
)
###########################
#### Inspect Results ####
###########################

###########
# Import data
setwd("C:/Users/Käyttäjä/Desktop/BDA/models/Results_weakinf_hierarchical")

# 1. Rolling-forecast results
results_df <- read.csv("results_forecast.csv", stringsAsFactors = FALSE)
results_df$Date <- as.Date(results_df$Date)   # convert back to Date

# 2. Convergence diagnostics over time
diagnostics_df <- read.csv("diagnostics_forecast.csv", stringsAsFactors = FALSE)
diagnostics_df$Date <- as.Date(diagnostics_df$Date)

# 3. Fitted model object (last model in the loop)
model <- readRDS("Weakinf_hierarchical_model.rds")
###########

# Overall RMSE and R-squared
overall_rmse <- sqrt(mean((results_df$Actual - results_df$Predicted)^2))
r_squared    <- cor(results_df$Actual, results_df$Predicted)^2
total_elpd   <- sum(results_df$Lpds)

cat("Overall RMSE:", overall_rmse, "\n")
cat("Overall R-squared:", r_squared, "\n")
cat("Overall ELPD:", total_elpd, "\n")

# Plot the results
ggplot(results_df, aes(x = Date)) +
  geom_line(aes(y = Predicted, colour = "Predicted")) +
  geom_line(aes(y = Actual,    colour = "Actual")) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "blue", alpha = 0.1) +
  labs(
    title = "Predicted vs Actual Returns",
    x = "Date",
    y = "Returns",
    colour = ""
  ) +
  theme_minimal()

#############################
#### Inspect diagnostics ####
#############################

summary(model)

param_diag_df <- bind_rows(diagnostics)

# Plot Rhats
ggplot(param_diag_df, aes(x = Date, y = Rhat)) +
  geom_line(color = "blue") +
  facet_wrap(~Parameter, scales = "free_y") +
  labs(
    title = "Rhat over time per parameter",
    x = "Date",
    y = "Rhat"
  ) +
  theme_minimal()

# Plot ESS
# Convert bulk and tail ESS to long format
ess_long <- param_diag_df %>%
  pivot_longer(
    cols = c(ESS_Bulk, ESS_Tail),
    names_to = "ESS_Type",
    values_to = "ESS_Value"
  )

ggplot(ess_long, aes(x = Date, y = ESS_Value, linetype = ESS_Type)) +
  geom_line(color = "blue") +                
  facet_wrap(~Parameter, scales = "free_y") +          
  labs(
    title = "ESS (Bulk and Tail) over time per parameter",
    x = "Date",
    y = "ESS",
    linetype = "Type"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
  )

mcmc_plot(model, type="trace")

# Posterior predictive checks
pp_check(model)
pp_check(model, type = "scatter_avg")
pp_check(model, type = "intervals")
pp_check(model, type = "error_hist")
pp_check(model, type = "error_scatter")


# Save the diagnostics and results
diagnostics_df <- bind_rows(diagnostics)

# as CSVs
write.csv(results_df,     "results_forecast.csv",     row.names = FALSE)
write.csv(diagnostics_df, "diagnostics_forecast.csv", row.names = FALSE)

# Save the model summary
smry <- summary(model)

# Save the printed version
capture.output(
  print(smry),
  file = "model_summary.txt"
)

# Save the model
saveRDS(model, file = "Weakinf_hierarchical_model.rds")



