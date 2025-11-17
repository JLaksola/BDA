
# Inflation bin function
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

# Train-only LOW/HIGH bins for GS10 (or any rate column)
make_rate_bins_LOW_HIGH <- function(train, df, col = "GS10") {
  thr <- stats::median(train[[col]], na.rm = TRUE)  # train-only threshold
  labs <- c("LOW","HIGH")
  out <- ifelse(df[[col]] <= thr, "LOW", "HIGH")
  factor(out, levels = labs)
}









