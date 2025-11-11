rm(list=ls())
library(lubridate)   
library(ggplot2)     
library(dplyr)      
library(brms)
library(cmdstanr)
library(purrr)


file_path <- "data/processed/Shiller_cleaned.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE) %>%
  mutate(
    Date = as.Date(Date),
    inv_CAPE = 1 / CAPE
  ) %>%
  filter(complete.cases(.)) %>%
  arrange(Date)

train_start <- as.Date("1881-01-01")
test_start  <- as.Date("1990-01-01")
test_end    <- as.Date("2015-02-01")

rmse_list   <- c()
predictions <- c()
actuals     <- c()
dates       <- as.Date(character())

# Generate monthly dates (like freq="MS" in pandas)
test_dates <- seq(from = test_start, to = test_end, by = "month")

# Function for rolling window inflation category calculation
inflation_cats <- function(df, end_train, window = 300){
  window_start <- end_train %m-% lubridate::years(window_years)
  train_window <- df %>% filter(Date >= window_start & Date <= end_train)
  
  previous_values <- train_window$Inflation
  end_train_value <- df %>% filter(Date == end_train) %>% pull(Inflation)
  
  positive_median <- median(previous_values[previous_values >= 0])
  if (end_train_value < 0){
    return ("negative")
  } else if (end_train_value < positive_median) {
    return ("low")
  } else {
    return ("high")
  }
}

# Function for generating posterior predictions
generate_prediction <- function(model, newdata){
  posterior_draws <- posterior_epred(
    model,
    newdata = newdata,
    allow_new_levels = TRUE
  )
  return (mean(posterior_draws))
}

# Formula for bayesian model
pooled_formula <- bf(
  Real_Return_10Y ~ 1 + (1 + CAPE | Inflation_Category),
  family = "gaussian",
  center = FALSE
)

priors <- c(
  prior(normal(0, 1), class = "b"),
  prior(normal(0, 1), class = "sd"),
  prior(lkj(2), class = "cor")
)

###########################
#### Fit Initial Model ####
###########################

initial_test_date <- as.Date(test_dates[1])
train_end <- initial_test_date %m-% lubridate::period(years = 10, months = 1)
train <- df %>% 
  filter(Date >= train_start, Date <= train_end)

model <- brm(
  formula = pooled_formula,
  prior = priors,
  data = train
)

##################################################
#### Rolling Forecast Loop With Model Updates ####
##################################################

for (i in seq_along(test_dates)) {
  
  current_date <- as.Date(test_dates[i])
  test_sample <- df %>%
    filter(Date == current_date) %>%
    mutate(Inflation_Category = inflation_cats(df, train_end))
  
  y_pred <- generate_prediction(model, test_sample)
  y_true <- test_sample$Real_Return_10Y
  rmse_i <- sqrt(mean((y_true - y_pred)^2))
  
  # Store results
  rmse_list   <- c(rmse_list, rmse_i)
  predictions <- c(predictions, as.numeric(y_pred))
  actuals     <- c(actuals, as.numeric(y_true))
  dates       <- c(dates, current_date)
  
  # New datapoint for training set
  train_end <- train_end %m+% lubridate::period(months = 1)
  train <- df %>%
    filter(Date >= train_start, Date <= train_end)
  
  # Update model
  model <- update(
    model,
    newdata = train,
    recompile = FALSE
  )
}

###########################
#### Inspect Results ####
###########################

# Convert results to data frame
results_df <- data.frame(
  Date      = dates,
  Predicted = predictions,
  Actual    = actuals,
  RMSE      = rmse_list
)

# Overall RMSE and R-squared
overall_rmse <- sqrt(mean((results_df$Actual - results_df$Predicted)^2))
r_squared    <- cor(results_df$Actual, results_df$Predicted)^2

cat("Overall RMSE:", overall_rmse, "\n")
cat("Overall R-squared:", r_squared, "\n")

# Plot the results
ggplot(results_df, aes(x = Date)) +
  geom_line(aes(y = Predicted, colour = "Predicted")) +
  geom_line(aes(y = Actual,    colour = "Actual")) +
  labs(
    title = "Predicted vs Actual Returns",
    x = "Date",
    y = "Returns",
    colour = ""
  ) +
  theme_minimal()

