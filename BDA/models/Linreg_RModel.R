# Load libraries
rm(list=ls())
library(lubridate)   # for date arithmetic
library(ggplot2)     # for plotting
library(dplyr)       # for data manipulation
library(brms)

# Load the data
file_path <- "C:/Users/antti/Desktop/Koulu/BDA/BDA/data/processed/Shiller_cleaned.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# Parse Date column and drop NAs
df$Date <- as.Date(df$Date) 
df <- df[complete.cases(df), ]

# Define the inverse CAPE variable
df$inv_CAPE <- 1 / df$CAPE


# Rolling forecast settings
train_start <- as.Date("1881-01-01")
test_start  <- as.Date("1990-01-01")
test_end    <- as.Date("2015-02-01")

# Initialize empty containers
rmse_list   <- c()
predictions <- c()
actuals     <- c()
dates       <- as.Date(character())

# Generate monthly dates (like freq="MS" in pandas)
test_dates <- seq(from = test_start, to = test_end, by = "month")

# Formula for bayesian model
pooled_formula <- bf(
  Real_Return_10Y ~ 1 + CAPE + Inflation,
  family = "gaussian",
  center = FALSE
)

# Function to return priors, priors could b adjusted as training proceeds?
priors <- function(){
  priors <- c(
    prior(normal(0, 1), class = "b", coef = "Intercept"),
    prior(normal(0, 1), class = "b", coef = "CAPE"),
    prior(normal(0, 1), class = "b", coef = "Inflation")
  )
  return (priors)
}

# Rolling Forecast Loop
for (i in seq_along(test_dates)) {
  current_date <- as.Date(test_dates[i])
  # Train end: 10 years and 1 month before current_date
  train_end <- current_date %m-% lubridate::period(years = 10, months = 1)
                                                   
  # Define train and test sets
  train <- df %>% 
    filter(Date >= train_start, Date <= train_end)
  test_sample <- df %>%
    filter(Date == current_date)
  
  if (nrow(test_sample) == 0 || nrow(train) == 0) next
  
  model <- brm(
    formula = pooled_formula,
    prior = priors(),
    data = train
  )
  
  y_pred <- posterior_epred(
    model,
    newdata = test_sample,
    allow_new_levels = TRUE
  )
  y_test <- test_sample$Real_Return_10Y
  
  rmse_i <- sqrt(mean((y_test - y_pred)^2))
  
  # Store results
  rmse_list   <- c(rmse_list, rmse_i)
  predictions <- c(predictions, as.numeric(y_pred))
  actuals     <- c(actuals, as.numeric(y_test))
  dates       <- c(dates, current_date)
}

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
