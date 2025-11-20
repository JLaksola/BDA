rm(list=ls())
library(lubridate)   
library(ggplot2)     
library(dplyr)      
library(brms)
library(purrr)
library(parallel)
library(posterior)
library(tidyr)
library(zoo)
source("~/Desktop/BDA/models/Functions.R")

# Preprocessing data
file_path <- "~/Desktop/BDA/data/processed/Shiller_cleaned.csv"
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
test_end    <- as.Date("2015-02-01")


########################
# Rolling 10-year average inflation
df$Inflation_10Y_Rolling <- zoo::rollmean(df$Inflation,
                                          k = 120,
                                          fill = NA,
                                          align = "right")


# Function for making inflation categories
inflation_cats <- function(past_values, current_value) {
  # Jos ei ole historiaa -> oletus "negative"
  if (length(past_values) == 0) {
    return("negative")
  }
  
  # Menneet positiiviset inflaatiot (poistetaan NA:t)
  pos_values <- past_values[!is.na(past_values) & past_values >= 0]
  
  # Jos ei ole menneitä positiivisia havaintoja
  if (length(pos_values) == 0) {
    if (is.na(current_value)) {
      return(NA_character_)
    }
    return(ifelse(current_value < 0, "negative", "low"))
  }
  
  # Jos nykyarvo on NA, palautetaan NA (voi halutessa muuttaa)
  if (is.na(current_value)) {
    return(NA_character_)
  }
  
  # Negatiiviset arvot omaan kategoriaan
  if (current_value < 0) {
    return("negative")
  }
  
  # Lasketaan 1/3- ja 2/3-quantile positiivisille
  qs <- quantile(pos_values, probs = c(1/3, 2/3), na.rm = TRUE)
  q1 <- qs[1]
  q2 <- qs[2]
  
  # Jaotellaan kolmeen yhtä suureen osaan positiiviset
  if (current_value < q1) {
    return("low")      # matala
  } else if (current_value < q2) {
    return("medium")   # keskitaso
  } else {
    return("high")     # korkea
  }
}

# 4. Pre-1980 ja 1980→ kategoriat -----------------------------------------

start_date <- as.Date("1990-01-01")

# Inflaatio rullaavana keskiarvona
inflation <- as.numeric(df$Inflation_10Y_Rolling)

# Maskit indeksin (Date-sarakkeen) perusteella
pre_mask  <- df$Date < start_date
post_mask <- !pre_mask

# Luodaan tyhjä kategoriasarake
cats <- rep(NA_character_, nrow(df))

# --- 4.1 Pre-1980: kategoriat koko pre-1980 jakson jakauman mukaan ---

# Pre-1980 positiiviset inflaatiot
pre_pos_values <- inflation[pre_mask & !is.na(inflation) & inflation >= 0]

if (length(pre_pos_values) > 0) {
  qs_pre <- quantile(pre_pos_values, probs = c(1/3, 2/3), na.rm = TRUE)
  q1_pre <- qs_pre[1]
  q2_pre <- qs_pre[2]
} else {
  q1_pre <- NA_real_
  q2_pre <- NA_real_
}

pre_indices <- which(pre_mask)

for (i in pre_indices) {
  val <- inflation[i]
  
  if (is.na(val)) {
    # halutessasi voit antaa esim. NA tai jonkin kategorian
    cats[i] <- NA_character_
  } else if (is.na(q1_pre)) {
    # fallback: jos ei ole positiivista dataa ennen 1980
    cats[i] <- ifelse(val < 0, "negative", "low")
  } else {
    if (val < 0) {
      cats[i] <- "negative"
    } else if (val < q1_pre) {
      cats[i] <- "low"
    } else if (val < q2_pre) {
      cats[i] <- "medium"
    } else {
      cats[i] <- "high"
    }
  }
}

# --- 4.2 Vuodesta 1980 alkaen: laajeneva ikkuna, joka sisältää myös pre-1980 datan ---

# Historiaksi kaikki pre-1980 inflaatiot (mukaan lukien NA:t – ne suodatetaan funktiossa)
past_values <- inflation[pre_mask]

post_indices <- which(post_mask)

for (i in post_indices) {
  val <- inflation[i]
  
  # Luokitus käyttäen KAIKKIA menneitä havaintoja (pre-1980 + 1980–t-1)
  cats[i] <- inflation_cats(past_values, val)
  
  # Päivitä historia (laajeneva ikkuna)
  past_values <- c(past_values, val)
}

# Lopuksi talteen dataan
df$Inflation_Category <- cats

# Tarkistus
tail(df[, c("Date", "Inflation", "Inflation_10Y_Rolling", "Inflation_Category")],100)

ggplot(df, aes(x = Date,
               y = Inflation_10Y_Rolling,
               color = Inflation_Category)) +
  geom_point(alpha = 0.6) +
  scale_color_manual(
    values = c(
      "negative" = "red",
      "low"      = "green",
      "medium"   = "orange",
      "high"     = "purple"
    )
  ) +
  labs(
    x = "Date",
    y = "Inflation",
    title = "Inflation over Time by Category",
    color = "Category"
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_line(),
    panel.grid.minor = element_line()
  )


##############

# Define inverse CAPE yield
df$Inv_CAPE = 1 / df$CAPE

# Look at summary and first rows
print(summary(df))
print(head(df))

# Define the target variable
target <- "Real_Return_10Y"

# Rolling forecast settings
train_start <- as.Date("1881-01-01")
test_start  <- as.Date("1960-01-01")
test_end    <- as.Date("2015-02-01")

# Initialize empty containers
rmse_list <- c()
predictions <- c()
actuals <- c()
medians <- c()
dates <- as.Date(character())

# Generate monthly dates (like freq="MS" in pandas)
test_dates <- seq(from = test_start, to = test_end, by = "month")

# Rolling Forecast Loop
for (i in seq_along(test_dates)) {
  # Make absolutely sure current_date is Date
  current_date <- as.Date(test_dates[i])
  
  # Train end: 10 years and 1 month before current_date
  train_end <- current_date %m-% lubridate::period(years = 10, months = 1)
  
  # Define train and test sets
  train <- df %>% 
    filter(Date >= train_start, Date <= train_end)
  
  test_sample <- df %>% filter(Date == current_date)
  
  # Skip if there is no test_sample for this date
  if (nrow(test_sample) == 0 || nrow(train) == 0) next
  
  # Create inflation categories
  train$Infl_Cat <- make_infl_bins_DEF_LOW_HIGH(train,train,"Inflation")
  test_sample$Infl_Cat <- make_infl_bins_DEF_LOW_HIGH(train,test_sample,"Inflation")
  
  # --- GS10 categories (LOW/HIGH) from TRAIN only ---
  train$GS10_Cat     <- make_rate_bins_LOW_HIGH(train, train, "GS10")
  test_sample$GS10_Cat <- make_rate_bins_LOW_HIGH(train, test_sample, "GS10")
  
  # Train linear model: Real_Return_10Y ~ CAPE
  model <- lm(
    Real_Return_10Y ~ CAPE + Inflation_Category,
    data = train
  )
  
  # Predict for the test sample
  y_pred <- predict(model, newdata = test_sample)
  y_test <- test_sample$Real_Return_10Y
  
  # RMSE for this single prediction
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
