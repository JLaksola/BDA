rm(list=ls())
library(dplyr)
library(ggplot2)
library(broom)

file_path <- "~/Desktop/BDA/data/processed/Shiller_cleaned.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE) %>%
  mutate(Date = as.Date(Date),
         Year = lubridate::year(Date),
         Period30_start = 1990 + 30 * ((Year - 1990) %/% 30),
         Period30_end   = Period30_start + 29,
         Period30       = factor(paste0(Period30_start, "-", Period30_end))) %>%
  filter(complete.cases(.)) %>%
  arrange(Date)

summary(df)
mean(df$CAPE)
sd(df$CAPE)
sd(df$Real_Return_10Y)

# Plot regression lines by 30-year period

ggplot(df, aes(x = CAPE, y = Real_Return_10Y)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE, colour = "red") +
  facet_wrap(~ Period30) +
  labs(
       x = "CAPE",
       y = "10-Year Ahead Annualized Real Returns (%)") +
  theme_bw()

# Inspect regression coefficients by 30-year period

reg_30 <- df %>%
  filter(!is.na(Period30)) %>%
  group_by(Period30) %>%
  do(
    tidy(lm(Real_Return_10Y ~ CAPE, data = .))
  )

reg_30
