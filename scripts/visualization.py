import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("outputs/predictions.csv")

# --------- 1. HISTOGRAMS OF RATIOS ----------
plt.figure(figsize=(10, 6))
df[["green_ratio", "yellow_ratio", "dark_ratio"]].hist(bins=20, figsize=(12, 6))
plt.suptitle("Distribution of Ripeness Color Ratios")
plt.show()

# --------- 2. BAR CHART OF LABEL COUNTS ----------
plt.figure(figsize=(6, 5))
df["pred_label"].value_counts().plot(kind="bar", color=["green", "yellow", "brown"])
plt.title("Count of Predicted Ripeness Labels")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# --------- 3. SCATTER PLOT: MEAN HUE vs MEAN VALUE ----------
plt.figure(figsize=(7, 6))
plt.scatter(df["mean_hue"], df["mean_value"], alpha=0.7)
plt.title("Mean Hue vs Mean Value")
plt.xlabel("Mean Hue")
plt.ylabel("Mean Value")
plt.show()

# --------- 4. SCATTER PLOT COLORED BY LABEL ----------
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="mean_hue", y="mean_value", hue="pred_label")
plt.title("Mean Hue vs Mean Value (Colored by Ripeness)")
plt.show()

# --------- 5. BOXPLOTS OF RATIOS PER LABEL ----------
plt.figure(figsize=(10, 6))
melt_df = df.melt(id_vars=["pred_label"], value_vars=["green_ratio", "yellow_ratio", "dark_ratio"])
sns.boxplot(data=melt_df, x="variable", y="value", hue="pred_label")
plt.title("Distribution of Color Ratios by Ripeness Label")
plt.xlabel("Ratio Type")
plt.ylabel("Value")
plt.show()

# --------- 6. CORRELATION HEATMAP ----------
plt.figure(figsize=(8, 6))
sns.heatmap(df[["green_ratio", "yellow_ratio", "dark_ratio", "mean_hue", "mean_value"]].corr(),
            annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
