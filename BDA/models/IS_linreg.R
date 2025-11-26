rm(list=ls())
library(dplyr)
library(ggplot2)
library(lubridate)


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

df <- df %>%
  mutate(
    Period = cut(
      Date,
      breaks = as.Date(c("1880-01-01","1980-01-01","2015-12-31")),
      labels = c("1880-1980","1980-2015"),
      right = FALSE
    )
  )

lm_list <- df %>%
  group_by(Period) %>%
  group_map(~ lm(Real_Return_10Y ~ CAPE, data = .x))

lm_full <- lm(Real_Return_10Y ~ CAPE * Period, data = df)


# 2A. One plot, all periods with separate regression lines
ggplot(df, aes(x = CAPE, y = Real_Return_10Y, colour = Period)) +
  geom_point(alpha = 0.4, size = 1) +
  geom_smooth(method = "lm", se = FALSE, size = 1.1) +
  labs(
    x = "CAPE",
    y = "10-year ahead real return (%)",
    title = "CAPE vs 10-year ahead returns by period"
  ) +
  theme_minimal()


# Lm with inflation categories
insample_start <- as.Date("1900-01-01")
insample_end   <- as.Date("2015-09-01")

insample_df <- df %>%
  filter(Date >= insample_start, Date <= insample_end)

lm_fit_cat <- lm(
  Real_Return_10Y ~ CAPE + Inflation_Category,
  data = insample_df
)

# Predictions
insample_df$Predicted <- predict(lm_fit_cat, newdata = insample_df)

# Simple RMSE / R^2
rmse_lm <- sqrt(mean((insample_df$Real_Return_10Y - insample_df$Predicted)^2))
r2_lm   <- cor(insample_df$Real_Return_10Y, insample_df$Predicted)^2
cat("OLS with categories RMSE:", rmse_lm, "\n")
cat("OLS with categories R^2:",  r2_lm, "\n")

# Plot
ggplot(insample_df, aes(x = Date)) +
  geom_line(aes(y = Real_Return_10Y, colour = "Actual")) +
  geom_line(aes(y = Predicted,      colour = "Predicted")) +
  labs(
    title = "In-sample OLS fit: CAPE + Inflation_Category",
    x = "Date",
    y = "10-year real return",
    colour = ""
  ) +
  theme_minimal()

# IS regression plot: returns vs CAPE, coloured by inflation category
ggplot(insample_df, aes(x = CAPE,
                        y = Real_Return_10Y,
                        colour = Inflation_Category)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    title = "In-sample regression: Real_Return_10Y ~ CAPE + Inflation_Category",
    x = "CAPE",
    y = "10-year real return"
  ) +
  theme_minimal()



