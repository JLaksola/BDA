## --- SETUP -------------------------------------------------------------
# df: your data frame
# df$Date: Date column
# df$Inflation: inflation rate (e.g. annualized %, numeric)

library(dplyr)
# Load the data
file_path <- "/Users/kayttaja/Desktop/BDA/data/processed/Shiller_cleaned.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# Parse Date column
df$Date <- as.Date(df$Date)          # assumes "YYYY-MM-DD" format
print(tail(df))
summary(df)

y <- df$Inflation                 # shorthand
n <- length(y)

p <- 12                             # AR order (AR(1); you can change to 2,3,…)
H <- 10 * 12                       # 10 years * 12 months
min_obs <- 60                      # require at least 60 observations before first fit

exp_infl_10y <- rep(NA_real_, n)   # will hold E_t[avg inflation next 10 years]
realized_infl_10y <- rep(NA_real_, n)
length(exp_infl_10y)

## --- ROLLING AR MODEL --------------------------------------------------
for (t in seq(min_obs, n - H)) {
  # 1. Use data up to time t (inclusive) as estimation sample
  y_est <- y[1:t]
  
  # 2. Fit AR(p) model to inflation
  #    We use an ARIMA with order (p,0,0), i.e. pure AR
  fit <- try(stats::arima(y_est, order = c(p, 0, 0)), silent = TRUE)
  if (inherits(fit, "try-error")) next  # skip if fit fails for some reason
  
  # 3. Forecast inflation H steps ahead
  fc <- predict(fit, n.ahead = H)$pred  # vector of length H
  
  # 4. Expected *average* inflation over the next 10 years
  exp_infl_10y[t] <- mean(fc)
}

## --- REALIZED 10-YEAR AVERAGE INFLATION --------------------------------
for (t in 1:(n - H)) {
  # realized inflation from t+1 to t+H
  realized_infl_10y[t] <- mean(y[(t + 1):(t + H)])
}

## --- ADD TO DATA FRAME -------------------------------------------------
df$ExpInfl10Y_AR      <- exp_infl_10y
df$RealizedInfl10Y    <- realized_infl_10y

plot_df <- df %>% filter(!is.na(ExpInfl10Y_AR), !is.na(RealizedInfl10Y))

# Example: quick check
plot(plot_df$Date, plot_df$ExpInfl10Y_AR, type = "l",
     xlab = "Date", ylab = "10-year average inflation",
     main = "Expected vs realized 10-year average inflation")
lines(plot_df$Date, plot_df$RealizedInfl10Y, lty = 2)
legend("topleft", legend = c("Expected (AR)", "Realized"),
       lty = c(1, 2), bty = "n")

print(tail(df,140))


