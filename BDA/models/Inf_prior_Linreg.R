rm(list = ls())
library(lubridate)
library(ggplot2)
library(dplyr)
library(brms)
library(purrr)
library(parallel)
library(posterior)
library(tidyr)
setwd("~/Desktop/BDA/models")
source("~/Desktop/BDA/models/Functions.R")

# Comment this out if cmdstan works
# install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/",getOption("repos")))
library(cmdstanr)
# dir.create(file.path("~/cmdstan"), showWarnings = FALSE)
# install_cmdstan(dir = "~/cmdstan", overwrite = TRUE)
set_cmdstan_path("~/cmdstan/cmdstan-2.37.0")

cores <- max(1, parallel::detectCores() - 1)

#########################
#### Preprocess Data ####
#########################

file_path <- "~/Desktop/BDA/data/processed/Shiller_cleaned.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE) %>%
  mutate(Date = as.Date(Date)) %>%
  filter(complete.cases(.)) %>%
  arrange(Date)

train_start <- as.Date("1881-01-01")
test_start  <- as.Date("1990-01-01")
test_end    <- as.Date("2015-09-01")

predictions <- c()
actuals     <- c()
lowers      <- c()
uppers      <- c()
dates_vec   <- as.Date(character())
diagnostics <- list()

# Generate monthly test dates
test_dates <- seq(from = test_start, to = test_end, by = "month")
n_iter     <- length(test_dates)

######################
#### Define Model ####
######################

formula <- bf(
  Real_Return_10Y ~ 1 + CAPE,
  family = gaussian(),
  center = FALSE
)

##################################################
#### Rolling Forecast Loop With Model Updates ####
##################################################

# --- Define prior update schedule (every 3 years) --------

prior_step_years <- 3L

prior_update_dates <- seq(
  from = test_start,
  to   = test_end,
  by   = paste0(prior_step_years, " years")
)

# list to store priors per block
priors_used <- vector("list", length(prior_update_dates))

# --- Main loop over prior blocks -------------------------

block_index <- 0

for (k in seq_along(prior_update_dates)) {
  
  prior_date <- prior_update_dates[k]
  
  # training end for the first test date in this prior block
  train_end <- prior_date %m-% years(10) %m-% months(1)
  train <- df %>%
    filter(Date >= train_start, Date <= train_end)
  
  current_priors <- build_priors_from_window(
    df           = df,
    train_end    = train_end,
    window_months = 360
  )
  
  # save priors + meta info for this block
  priors_used[[k]] <- list(
    block      = k,
    prior_date = prior_date,
    train_end  = train_end,
    priors     = current_priors
  )
  
  model <- brm(
    formula = formula,
    prior   = current_priors,
    data    = train,
    chains  = 3,
    iter    = 2000,
    warmup  = 1000,
    cores   = cores,
    backend = "cmdstanr"
  )
  
  # block boundaries
  block_start <- prior_date
  block_end   <- if (k < length(prior_update_dates)) {
    prior_update_dates[k + 1] %m-% months(1)
  } else {
    test_end
  }
  
  block_dates <- test_dates[test_dates >= block_start & test_dates <= block_end]
  
  # ---- inner monthly loop ----
  for (current_date in block_dates) {
    
    block_index <- block_index + 1
    cat("\nBLOCK", k, "STEP", block_index,
        "DATE", as.character(current_date), "\n")
    
    # Test sample for this month
    test_sample <- df %>% filter(Date == current_date)
    
    # --- Prediction ---
    posterior_draws <- posterior_epred(
      model,
      newdata         = test_sample,
      allow_new_levels = TRUE
    )
    y_pred   <- mean(posterior_draws)
    ci_lower <- quantile(posterior_draws, probs = 0.025)
    ci_upper <- quantile(posterior_draws, probs = 0.975)
    y_true   <- test_sample$Real_Return_10Y
    
    predictions <- c(predictions, as.numeric(y_pred))
    actuals     <- c(actuals, as.numeric(y_true))
    lowers      <- c(lowers, ci_lower)
    uppers      <- c(uppers, ci_upper)
    dates_vec   <- c(dates_vec, current_date)
    
    # Store convergence diagnostics
    diag_df <- convergence_diagnostics(current_date, model)
    diag_df$Block <- k
    diag_df$Step  <- block_index
    
    diagnostics[[block_index]] <- diag_df
    
    # Posterior predictive check at the end of the block
    if (current_date == block_end) {
      png(paste0("pp_check_block_", k, "_", current_date, ".png"),
          width = 800, height = 600)
      pp_check(model, ndraws = 50)
      dev.off()
    }
    
    # --- Update training data for next month in this block ---
    next_train_end <- current_date %m-% years(10) %m-% months(1)
    
    if (next_train_end > train_end) {
      train_end <- next_train_end
      train <- df %>%
        filter(Date >= train_start, Date <= train_end)
      
      model <- update(
        model,
        newdata   = train,
        recompile = FALSE
      )
    }
  }
}

###########################
#### Inspect Results ####
###########################

results_df <- data.frame(
  Date      = dates_vec,
  Predicted = predictions,
  Actual    = actuals,
  Upper     = uppers,
  Lower     = lowers
)

overall_rmse <- sqrt(mean((results_df$Actual - results_df$Predicted)^2))
r_squared    <- cor(results_df$Actual, results_df$Predicted)^2

cat("Overall RMSE:", overall_rmse, "\n")
cat("Overall R-squared:", r_squared, "\n")

ggplot(results_df, aes(x = Date)) +
  geom_line(aes(y = Predicted, colour = "Predicted")) +
  geom_line(aes(y = Actual,    colour = "Actual")) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper),
              alpha = 0.1) +
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

ggplot(param_diag_df, aes(x = Date, y = Rhat)) +
  geom_line() +
  facet_wrap(~ Parameter, scales = "free_y") +
  labs(
    title = "Rhat over time per parameter",
    x = "Date",
    y = "Rhat"
  ) +
  theme_minimal()

ess_long <- param_diag_df %>%
  pivot_longer(
    cols      = c(ESS_Bulk, ESS_Tail),
    names_to  = "ESS_Type",
    values_to = "ESS_Value"
  )

ggplot(ess_long, aes(x = Date, y = ESS_Value, linetype = ESS_Type)) +
  geom_line() +
  facet_wrap(~ Parameter, scales = "free_y") +
  labs(
    title = "ESS (Bulk and Tail) over time per parameter",
    x = "Date",
    y = "ESS",
    linetype = "Type"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

mcmc_plot(model, type = "trace")



