import pandas as pd

# -------------------------------
# LOAD DATA
# -------------------------------
FILE_PATH = "results.csv"   # adjust if needed

df = pd.read_csv(FILE_PATH)

df.columns = [
    "Config ID", "Dataset",
    "Window","K" ,"Hidden Size", "Num Layers",
    "Ridge Alpha", "Input Scaling",
    "RMSE", "MAE", "MAPE", "Training Time"
]

print("\nLoaded data:", df.shape)


# -------------------------------
# CONFIG-LEVEL AGGREGATION
# -------------------------------
config_perf = (
    df.groupby("Config ID")
    .agg({
        "Window": "first",
        "K": "first",
        "Hidden Size": "first",
        "Num Layers": "first",
        "Ridge Alpha": "first",
        "Input Scaling": "first",
        "RMSE": "mean",
        "MAE": "mean",
        "MAPE": "mean",
        "Training Time": "mean"
    })
    .reset_index()
)

print("\nTotal configs:", len(config_perf))


# -------------------------------
# TOP CONFIGS
# -------------------------------
TOP_K = 10

top_configs = config_perf.sort_values("RMSE").head(TOP_K)

print("\n==============================")
print("TOP CONFIGS (by RMSE)")
print("==============================")
print(top_configs)


# -------------------------------
# HYPERPARAMETER DISTRIBUTION (TOP CONFIGS)
# -------------------------------
print("\n==============================")
print("TOP CONFIG HYPERPARAMETER DISTRIBUTION")
print("==============================")

for col in ["Window", "K", "Hidden Size", "Num Layers", "Ridge Alpha", "Input Scaling"]:
    print(f"\n{col} distribution:")
    print(top_configs[col].value_counts())


# -------------------------------
# GLOBAL SENSITIVITY ANALYSIS
# -------------------------------
print("\n==============================")
print("GLOBAL HYPERPARAMETER SENSITIVITY (AVG RMSE)")
print("==============================")

def print_group_analysis(column):
    grouped = df.groupby(column)["RMSE"].mean().sort_values()
    print(f"\n{column}:")
    print(grouped)

for col in ["Window", "K", "Hidden Size", "Num Layers", "Ridge Alpha", "Input Scaling"]:
    print_group_analysis(col)


# -------------------------------
# INTERACTION EFFECTS
# -------------------------------
print("\n==============================")
print("INTERACTION: Num Layers × Input Scaling")
print("==============================")

interaction = (
    df.groupby(["Num Layers", "Input Scaling"])["RMSE"]
    .mean()
    .unstack()
)

print(interaction)


# -------------------------------
# GOOD vs BAD REGION ANALYSIS
# -------------------------------
print("\n==============================")
print("GOOD vs BAD CONFIG REGIONS")
print("==============================")

# Define thresholds (adjust if needed)
good_threshold = config_perf["RMSE"].quantile(0.1)
bad_threshold  = config_perf["RMSE"].quantile(0.9)

good_configs = config_perf[config_perf["RMSE"] <= good_threshold]
bad_configs  = config_perf[config_perf["RMSE"] >= bad_threshold]

print("\nGOOD CONFIG REGION (Top 10%)")
print(good_configs.describe())

print("\nBAD CONFIG REGION (Bottom 10%)")
print(bad_configs.describe())


# -------------------------------
# BEST CONFIG (FINAL)
# -------------------------------
best_config = top_configs.iloc[0]

print("\n==============================")
print("BEST CONFIG")
print("==============================")
print(best_config)


# -------------------------------
# SAVE RESULTS
# -------------------------------
top_configs.to_csv("top_10_configs.csv", index=False)
config_perf.to_csv("config_performance.csv", index=False)

print("\nSaved:")
print(" - top_10_configs.csv")
print(" - config_performance.csv")