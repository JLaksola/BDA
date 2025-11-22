library(dplyr)
library(lubridate)
library(brms)

#### Inflation bin function
make_infl_bins_DEF_LOW_HIGH <- function(train, df, col) {
  xtr  <- train[[col]]
  # median of strictly positive inflation in TRAIN
  pos  <- xtr[xtr > 0 & is.finite(xtr)]
  med  <- if (length(pos) > 0) stats::median(pos, na.rm = TRUE) else NA_real_
  # tiny epsilon to ensure strictly increasing breaks if needed
  eps  <- 1e-12
  if (!is.finite(med) || med <= 0) med <- 0 + eps
  
  # (-Inf, 0] = NEG, (0, med] = LOW, (med, Inf] = HIGH
  edges  <- c(-Inf, 0, med, Inf)
  labels <- c("NEG", "LOW", "HIGH")
  
  binned <- cut(df[[col]], breaks = edges, labels = labels,
                include.lowest = TRUE, right = TRUE)
  # enforce fixed levels even if some bins are empty
  factor(binned, levels = labels)
}

#### Train-only LOW/HIGH bins for GS10 (or any rate column)
make_rate_bins_LOW_HIGH <- function(train, df, col = "GS10") {
  thr <- stats::median(train[[col]], na.rm = TRUE)  # train-only threshold
  labs <- c("LOW","HIGH")
  out <- ifelse(df[[col]] <= thr, "LOW", "HIGH")
  factor(out, levels = labs)
}


#### Build priors
build_priors_from_window <- function(df, train_end, window_months = 360) {
  # Start of prior window: 360 months before train_end
  prior_start <- train_end %m-% months(window_months)
  
  prior_window <- df %>%
    filter(Date > prior_start, Date <= train_end)
  
  if (nrow(prior_window) < 50) {
    stop("Not enough data in prior_window to estimate priors.")
  }
  
  # Classical linear regression on the prior window
  lm_fit <- lm(Real_Return_10Y ~ CAPE, data = prior_window)
  summ <- summary(lm_fit)
  
  beta_tab <- summ$coefficients
  
  intercept_mean <- beta_tab["(Intercept)", "Estimate"]
  intercept_sd <- beta_tab["(Intercept)", "Std. Error"]
  
  slope_mean <- beta_tab["CAPE", "Estimate"]
  slope_sd <- beta_tab["CAPE", "Std. Error"]
  
  sigma_hat <- summ$sigma  # residual sd from OLS
  
  priors <- c(
    prior(normal(intercept_mean, intercept_sd), class = "b", coef = "Intercept"),
    prior(normal(slope_mean, slope_sd), class = "b", coef = "CAPE"),
    # sigma prior: centered around 0 with scale sigma_hat, truncated > 0 by brms
    prior(student_t(3, 0, sigma_hat), class = "sigma")
  )
  
  return(priors)
}

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
    print(par_draws)
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










