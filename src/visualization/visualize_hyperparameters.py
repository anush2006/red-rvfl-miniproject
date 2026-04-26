import pandas as pd
import matplotlib.pyplot as plt
import os


INPUT_FILES = {
    "K": "hyperparameter_isolates/k_isolate.csv",
    "Layers": "hyperparameter_isolates/layer_isolate.csv",
    "Hidden": "hyperparameter_isolates/hidden_isolate.csv",
    "Scaling": "hyperparameter_isolates/scaling_isolate.csv",
    "Window": "hyperparameter_isolates/window_isolate.csv",
    "Ridge": "hyperparameter_isolates/ridge_isolate.csv",
}

OUTPUT_DIR = "ablation_plots/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_df(file, param_name):
    df = pd.read_csv(file)

    # average across datasets if needed
    if "Dataset" in df.columns:
        df = df.groupby(param_name)["RMSE"].mean().reset_index()

    df = df.sort_values(param_name)
    return df


def plot_isolate(file, param_name):
    df = process_df(file, param_name)

    plt.figure()
    plt.plot(df[param_name], df["RMSE"], marker='o')

    plt.title(f"RMSE vs {param_name} (Isolated)")
    plt.xlabel(param_name)
    plt.ylabel("RMSE")

    filename = f"{param_name}_isolate.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

    print(f"Saved: {filename}")

    print(f"\n{param_name} variation:")
    print("Min RMSE:", df["RMSE"].min())
    print("Max RMSE:", df["RMSE"].max())
    print("Diff:", df["RMSE"].max() - df["RMSE"].min())

def plot_combined_importance(input_files):
    summary = []

    for param, file in input_files.items():

        if not os.path.exists(file):
            print(f"Missing (skipped in combined): {file}")
            continue

        df = process_df(file, param)

        min_rmse = df["RMSE"].min()
        max_rmse = df["RMSE"].max()

        diff = max_rmse - min_rmse

        summary.append({
            "Parameter": param,
            "Impact": diff
        })

    summary_df = pd.DataFrame(summary)

    # sort by importance
    summary_df = summary_df.sort_values("Impact", ascending=False)

    # ---- Plot ----
    plt.figure()
    plt.bar(summary_df["Parameter"], summary_df["Impact"])

    plt.title("Absolute Impact of Hyperparameters on RMSE")
    plt.xlabel("Hyperparameter")
    plt.ylabel("RMSE Change")

    plt.savefig(os.path.join(OUTPUT_DIR, "combined_importance.png"))
    plt.close()

    print("\nCombined Importance:")
    print(summary_df)


# -------------------------------
# MAIN LOOP
# -------------------------------
for param, file in INPUT_FILES.items():

    if not os.path.exists(file):
        print(f"Missing: {file}")
        continue

    print(f"\nProcessing {param}")
    plot_isolate(file, param)


# Combined plot
plot_combined_importance(INPUT_FILES)


print("\nDone. Check:", OUTPUT_DIR)