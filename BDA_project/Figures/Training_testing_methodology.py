import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

train_start = pd.Timestamp("1881-12-01")
test_start = pd.Timestamp("1990-01-01")

step_size_months = 6
val_window_months = 6
gap_years = 10
gap_months = 1
n_folds = 5

rows = []
for fold in range(1, n_folds + 1):
    # validation window
    val_start = test_start + pd.DateOffset(months=(fold - 1) * step_size_months)
    val_end = val_start + pd.DateOffset(months=val_window_months - 1)

    # training window ends 10y + 1m before validation window end
    train_end = val_end - pd.DateOffset(years=gap_years, months=gap_months)

    rows.append(
        {
            "fold": fold,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
        }
    )

splits_df = pd.DataFrame(rows)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 4))

for i, row in splits_df.iterrows():
    y = row["fold"]

    # train segment
    ax.hlines(
        y,
        xmin=row["train_start"],
        xmax=row["train_end"],
        color="blue",
        linewidth=8,
        label="Train" if i == 0 else "",
    )

    # validation segment
    ax.hlines(
        y,
        xmin=row["val_start"],
        xmax=row["val_end"],
        color="orange",
        linewidth=8,
        label="Test" if i == 0 else "",
    )

ax.set_yticks(splits_df["fold"])
ax.set_yticklabels([f"Fold {f}" for f in splits_df["fold"]])

ax.xaxis.set_major_locator(mdates.YearLocator(base=10))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

ax.set_xlabel("Date")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
