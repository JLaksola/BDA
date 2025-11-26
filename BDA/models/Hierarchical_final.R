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
  mutate(Date = as.Date(Date),
         Year = lubridate::year(Date),
         Period30_start = 1990 + 30 * ((Year - 1990) %/% 30),
         Period30_end   = Period30_start + 29,
         Period30       = factor(paste0(Period30_start, "-", Period30_end)),
         Period40_start = 1980 + 40 * ((Year - 1980) %/% 40),
         Period40_end   = Period40_start + 39,
         Period40       = factor(paste0(Period40_start, "-", Period40_end))) %>%
  filter(complete.cases(.)) %>%
  arrange(Date)

train_start <- as.Date("1881-01-01")
test_start  <- as.Date("1990-01-01")
test_end    <- as.Date("2015-09-01")

predictions <- c()
actuals     <- c()
lowers      <- c()
uppers      <- c()
lpds         <- c()
dates_vec   <- as.Date(character())
diagnostics <- list()

# Generate monthly test dates
test_dates <- seq(from = test_start, to = test_end, by = "month")
n_iter     <- length(test_dates)
step_size_months <- 6L

######################
#### Define Model ####
######################

formula <- bf(
  Real_Return_10Y ~ 1 + CAPE + (1 + CAPE | Period40),
  family = "gaussian",
  center = FALSE
)
get_prior(formula,data=df)

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
  
  prior_date <- as.Date(prior_update_dates[k])
  
  # training end for the first test date in this prior block
  train_end <- as.Date(prior_date) %m-% years(10) %m-% months(1)
  train <- df %>%
    filter(Date >= train_start, Date <= train_end)
  
  current_priors <- hierarchical_priors_from_window_uncentered(
    df            = df,
    train_end     = train_end,
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
    backend = "cmdstanr",
    cores   = cores,
    adapt_delta   = 0.99,
    max_treedepth = 15,
    seed    = 1
  )
  
  
  # block boundaries
  block_start <- as.Date(prior_date)
  block_end   <- if (k < length(prior_update_dates)) {
    as.Date(prior_update_dates[k + 1]) %m-% months(1)
  } else {
    test_end
  }
  
  block_dates <- seq(
    from = block_start,
    to   = block_end,
    by   = paste0(step_size_months, " months")
  )
  
  # ---- inner monthly loop ----
  for (current_date in block_dates) {
    
    current_date <- as.Date(current_date)
    
    block_index <- block_index + 1
    cat("\nBLOCK", k, "STEP", block_index,
        "DATE", as.character(current_date), "\n")
    
    ## ---- define a 6-month test window ----
    window_end <- min(current_date %m+% months(step_size_months - 1L),
                      block_end)
    
    test_sample <- df %>%
      filter(Date >= current_date, Date <= window_end)
    
    if (nrow(test_sample) == 0) next  # just in case
    
    ## ---- vectorised prediction for all rows in test_sample ----
    posterior_draws <- posterior_epred(
      model,
      newdata          = test_sample,
      allow_new_levels = TRUE
    )
    # posterior_draws: iterations x n_test
    
    y_pred_vec   <- colMeans(posterior_draws)
    ci_lower_vec <- apply(posterior_draws, 2, quantile, probs = 0.025)
    ci_upper_vec <- apply(posterior_draws, 2, quantile, probs = 0.975)
    y_true_vec   <- test_sample$Real_Return_10Y
    
    # log predictive density for each row
    lpd_vec <- compute_log_pred_density_hier(model, test_sample)
    
    ## ---- store results for ALL rows in this 6-month window ----
    predictions <- c(predictions, as.numeric(y_pred_vec))
    actuals     <- c(actuals,   as.numeric(y_true_vec))
    lowers      <- c(lowers,    ci_lower_vec)
    uppers      <- c(uppers,    ci_upper_vec)
    dates_vec   <- c(dates_vec, test_sample$Date)
    lpds        <- c(lpds,      lpd_vec)
    
    ## ---- convergence diagnostics once per step (still per block) ----
    diag_df <- convergence_diagnostics(current_date, model)
    diag_df$Block <- k
    diag_df$Step  <- block_index
    diagnostics[[block_index]] <- diag_df
    
    ## ---- update training end only after this window ----
    next_train_end <- window_end %m-% years(10) %m-% months(1)
    
    if (next_train_end > train_end) {
      train_end <- next_train_end
      train <- df %>%
        filter(Date >= train_start, Date <= train_end)
      
      model <- update(
        model,
        newdata   = train,
        recompile = FALSE)
    }
  }
  saveRDS(
    model,
    file = paste0(
      "Pooled_weakinf_block_",
      sprintf("%02d", k), "_",
      format(block_end, "%Y-%m-%d"),
      ".rds"
    )
  )
}



###########################
#### Inspect Results ####
###########################

results_df <- data.frame(
  Date      = dates_vec,
  Predicted = predictions,
  Actual    = actuals,
  Upper     = uppers,
  Lower     = lowers,
  Lpds      = lpds
)

overall_rmse <- sqrt(mean((results_df$Actual - results_df$Predicted)^2))
r_squared    <- cor(results_df$Actual, results_df$Predicted)^2
total_elpd   <- sum(lpds)

cat("Overall RMSE:", overall_rmse, "\n")
cat("Overall R-squared:", r_squared, "\n")
cat("Overall ELPD:", total_elpd, "\n")

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

# Save the model priors
# Flatten priors_used into a single data frame
priors_df <- map_dfr(priors_used, function(x) {
  if (is.null(x)) return(NULL)  # in case some entries are empty
  
  p_df <- as.data.frame(x$priors)  # brmsprior -> data frame
  
  # Add block-level info to each prior row
  p_df$block      <- x$block
  p_df$prior_date <- x$prior_date
  p_df$train_end  <- x$train_end
  
  p_df
})

# Optional: reorder columns a bit
priors_df <- priors_df %>%
  select(block, prior_date, train_end, everything())

# Save to CSV
write.csv(priors_df, "priors_used.csv", row.names = FALSE)