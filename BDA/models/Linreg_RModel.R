rm(list=ls())
library(lubridate)   
library(ggplot2)     
library(dplyr)      
library(brms)
library(purrr)
library(cmdstanr)
library(parallel)
library(posterior)
library(tidyr)

# Comment this out if cmdstan works
#dir.create(file.path("~/cmdstan"), showWarnings = FALSE)
#install_cmdstan(dir = "~/cmdstan")
set_cmdstan_path("~/cmdstan/cmdstan-2.37.0")

cores <- max(1, parallel::detectCores() - 1)

#########################
#### Preprocess Data ####
#########################

file_path <- "data/processed/Shiller_cleaned.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE) %>%
  mutate(
    Date = as.Date(Date),
    inv_CAPE = 1 / CAPE,
    CAPE_scaled = scale(CAPE),
    Inflation_scaled = scale(Inflation),
    Real_Return_10Y_scaled = scale(Real_Return_10Y)
  ) %>%
  filter(complete.cases(.)) %>%
  arrange(Date)

train_start <- as.Date("1881-01-01")
test_start  <- as.Date("1990-01-01")
test_end    <- as.Date("2015-02-01")

rmse_list     <- c()
predictions   <- c()
actuals       <- c()
lowers        <- c()
uppers        <- c()
lpds         <- c()
dates         <- as.Date(character())
diagnostics   <- list()

# Generate monthly dates
test_dates <- seq(from = test_start, to = test_end, by = "month")
n_iter <- length(test_dates)

# Function for rolling window inflation category calculation
inflation_cats <- function(past_values, current_value){
  # First datapoint negative
  if (length(past_values) == 0) {
    return("negative")
  }
  pos_values <- past_values[past_values >= 0]
  pos_median <- if(length(pos_values) == 0) 0 else median(pos_values)
  if (current_value < 0) {
    return("negative")
  } else if (current_value < pos_median) {
    return("low")
  } else {
    return("high")
  }
}

# Compute inflation categories
window_size <- 120
df <- df %>%
  mutate(
    Inflation_Category = map_chr(
      seq_len(nrow(df)),
      ~ {
        hist_values <- df$Inflation[max(1, .x - window_size):(.x - 1)]
        inflation_cats(hist_values, df$Inflation[.x])
      }
    ),
    Inflation_Category = factor(
      Inflation_Category,
      levels = c("negative", "low", "high")
    )
  )

###########################
#### Utility functions ####
###########################

# Function for generating posterior predictions
generate_prediction <- function(model, newdata, prob = 0.95){
  posterior_draws <- posterior_epred(
    model,
    newdata = newdata,
    allow_new_levels = TRUE
  )
  pred_mean <- mean(posterior_draws)
  ci_lower <- quantile(posterior_draws, probs = (1 - prob) / 2)
  ci_upper <- quantile(posterior_draws, probs = 1 - (1 - prob) / 2)
  return (c(pred_mean, ci_lower, ci_upper))
}

# Store convergence diagnostics
convergence_diagnostics <- function(current_date, model){
  # Posterior parameter draws
  draws_array <- as_draws_array(model)
  
  # Automatically select parameter columns (exclude metadata/internal columns)
  param_cols <- setdiff(dimnames(draws_array)[[3]], c("lp__", "lprior"))
  
  # Initialize vectors to store diagnostics
  rhat_vals <- numeric(length(param_cols))
  ess_bulk_vals <- numeric(length(param_cols))
  ess_tail_vals <- numeric(length(param_cols))
  
  # Loop over parameters
  for (i in seq_along(param_cols)) {
    par <- param_cols[i]
    par_draws <- draws_array[,,par]  # iterations x chains
    rhat_vals[i] <- rhat(par_draws)
    ess_bulk_vals[i] <- ess_bulk(par_draws)
    ess_tail_vals[i] <- ess_tail(par_draws)
  }
  
  # Combine into data frame
  diag_df <- data.frame(
    Date      = current_date,
    Parameter = param_cols,
    Rhat      = rhat_vals,
    ESS_Bulk  = ess_bulk_vals,
    ESS_Tail  = ess_tail_vals
  )
  
  return (diag_df)
}

# Compute lpd
compute_log_pred_density <- function(model, newdata) {
  # 1. Get the log-likelihood for the new data point(s) given each posterior sample.
  # This returns a matrix of dimensions (number of posterior samples) x (number of data points)
  log_lik_matrix <- log_lik(model, newdata = newdata, allow_new_levels = TRUE)
  
  # 2. Convert log-likelihoods to likelihoods (p(y_i|theta^{(s)}))
  likelihoods_matrix <- exp(log_lik_matrix)
  
  # 3. Calculate the LPD for each data point: log( (1/S) * sum(likelihoods) )
  # We use the 'colMeans' to get the mean of the likelihoods across all posterior samples (rows)
  # Then we take the log.
  log_pred_density <- log(colMeans(likelihoods_matrix))
  
  # log_pred_density is a vector, one LPD for each row in newdata.
  # Since your loop only uses one test sample (row) at a time, we return the first element.
  return(log_pred_density[1])
}

######################
#### Define Model ####
######################

# Formula for bayesian model
formula <- bf(
  Real_Return_10Y ~ 1 + CAPE_scaled + (1 + CAPE_scaled | Inflation_Category),
  family = "gaussian",
  center = FALSE
)

priors <- c(
  prior(normal(0, 1), coef = "Intercept"),    
  prior(normal(0, 0.5), class = "b", coef = "CAPE_scaled"),
  prior(normal(0, 3), class = "sd", group = "Inflation_Category", coef = "Intercept"), 
  prior(normal(0, 3), class = "sd", group = "Inflation_Category", coef = "CAPE_scaled"),
  prior(lkj(2), class = "cor"),
  prior(student_t(3, 0, 1), class = "sigma")
)

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
  cores = cores,
  adapt_delta = 0.99,
  max_treedepth = 20,
  seed = 1
)

##################################################
#### Rolling Forecast Loop With Model Updates ####
##################################################

for (i in seq_along(test_dates)) {
  
  if (i %% 1 != 0){
    next
  }
  
  if (i > 1){
    break
  }
  
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
  rmse_i <- sqrt(mean((y_true - y_pred)^2))
  lpd <- compute_log_pred_density(model, test_sample)
  
  # Store results
  rmse_list     <- c(rmse_list, rmse_i)
  predictions   <- c(predictions, as.numeric(y_pred))
  actuals       <- c(actuals, as.numeric(y_true))
  lowers        <- c(lowers, lower)
  uppers        <- c(uppers, upper)
  dates         <- c(dates, current_date)
  lpds         <- c(lpds, lpd)
  
  # Store convergence diagnostics
  diagnostics[[i]] <- convergence_diagnostics(current_date, model)
  
  # New datapoint for training set
  train_end <- train_end %m+% lubridate::period(months = 1)
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
  Date         = dates,
  Predicted    = predictions,
  Actual       = actuals,
  Upper        = uppers,
  Lower        = lowers,
  RMSE         = rmse_list
)

###########################
#### Inspect Results ####
###########################

# Overall RMSE and R-squared
overall_rmse <- sqrt(mean((results_df$Actual - results_df$Predicted)^2))
r_squared    <- cor(results_df$Actual, results_df$Predicted)^2
total_elpd   <- sum(lpds)

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
pp_check(model, type = "dens_overlay_grouped", group = "Inflation_Category")