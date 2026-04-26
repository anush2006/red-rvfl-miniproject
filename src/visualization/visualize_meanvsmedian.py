import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. LOAD DATA
df_mean = pd.read_csv('mean-aggregation/config_performance_mean.csv')
top_mean = pd.read_csv('mean-aggregation/top_10_configs_mean.csv').iloc[0]
top_median = pd.read_csv('median-aggregation/top_10_configs_median.csv').iloc[0]

# 2. CREATE HEATMAPS
def create_heatmaps(df):
    plt.figure(figsize=(14, 5))
    
    # Heatmap A: Complexity Plateau (Hidden Size vs Layers)
    # We fix the best performing Window (48), Ridge (0.01), and Scaling (0.5)
    plateau_data = df[(df['Window'] == 48) & 
                      (df['Ridge Alpha'] == 0.01) & 
                      (df['Input Scaling'] == 0.5)]
    pivot_plateau = plateau_data.pivot(index="Num Layers", columns="Hidden Size", values="RMSE")

    plt.subplot(1, 2, 1)
    sns.heatmap(pivot_plateau, annot=True, fmt=".2f", cmap="YlGn", cbar_kws={'label': 'RMSE'})
    plt.title("Complexity Plateau\n(Fixed: Window=48, λ=0.01, Scale=0.5)")

    # Heatmap B: Sensitivity Analysis (Ridge Alpha vs Input Scaling)
    # We fix Window=48, Layers=1, Hidden=50 (the efficient baseline)
    sensitivity_data = df[(df['Window'] == 48) & 
                          (df['Num Layers'] == 1) & 
                          (df['Hidden Size'] == 50)]
    pivot_sensitivity = sensitivity_data.pivot(index="Ridge Alpha", columns="Input Scaling", values="RMSE")

    plt.subplot(1, 2, 2)
    # Using a diverging map to highlight the "danger zone" at higher Alphas
    sns.heatmap(pivot_sensitivity, annot=True, fmt=".2f", cmap="RdYlGn_r", cbar_kws={'label': 'RMSE'})
    plt.title("Hyperparameter Sensitivity\n(Fixed: Window=48, Layers=1, Hidden=50)")

    plt.tight_layout()
    plt.savefig('rvfl_heatmaps.png', dpi=300)
    plt.show()

# 3. CREATE RADAR CHART (Mean vs Median)
def create_radar(top_mean, top_median):
    categories = ['Window', 'Hidden Size', 'Num Layers', 'Ridge Alpha', 'Input Scaling']
    
    # Normalization ranges based on your grid
    ranges = {
        'Window': (48, 124),
        'Hidden Size': (20, 200),
        'Num Layers': (1, 7),
        'Ridge Alpha': (0.001, 1.0), # Log scale for the chart logic
        'Input Scaling': (0.1, 1.0)
    }

    def normalize(row):
        vals = []
        vals.append((row['Window'] - 48) / (124 - 48))
        vals.append((row['Hidden Size'] - 20) / (200 - 20))
        vals.append((row['Num Layers'] - 1) / (7 - 1))
        # Log scale for Ridge Alpha as it spans magnitudes
        vals.append((np.log10(row['Ridge Alpha']) - np.log10(0.001)) / (np.log10(1.0) - np.log10(0.001)))
        vals.append((row['Input Scaling'] - 0.1) / (1.0 - 0.1))
        return vals

    mean_coords = normalize(top_mean)
    median_coords = normalize(top_median)

    # Close the loop
    mean_coords += mean_coords[:1]
    median_coords += median_coords[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.plot(angles, mean_coords, color='#1f77b4', linewidth=2, label='Top Mean Config')
    ax.fill(angles, mean_coords, color='#1f77b4', alpha=0.25)
    
    ax.plot(angles, median_coords, color='#ff7f0e', linewidth=2, linestyle='--', label='Top Median Config')
    ax.fill(angles, median_coords, color='#ff7f0e', alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    plt.title("Hyperparameter Signature: Mean vs Median Top Configs", y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.savefig('rvfl_radar_comparison.png', dpi=300)
    plt.show()

# Run the functions
create_heatmaps(df_mean)
create_radar(top_mean, top_median)# 2. HEATMAPS (FIXED)
plt.figure(figsize=(14, 6))

# Subplot 1: Complexity Plateau
df_plateau = df_mean[(df_mean['Window'] == 48) & 
                     (df_mean['Ridge Alpha'] == 0.01) & 
                     (df_mean['Input Scaling'] == 0.5)]
pivot_plateau = df_plateau.pivot(index="Num Layers", columns="Hidden Size", values="RMSE")

plt.subplot(1, 2, 1)
sns.heatmap(pivot_plateau, annot=True, fmt=".2f", cmap="YlGn", cbar_kws={'label': 'RMSE'})
plt.title("Complexity Plateau\n(Fixed: Window=48, λ=0.01, Scale=0.5)", pad=20)

# Subplot 2: Sensitivity Analysis
df_sensitivity = df_mean[(df_mean['Window'] == 48) & 
                          (df_mean['Num Layers'] == 1) & 
                          (df_mean['Hidden Size'] == 50)]
pivot_sensitivity = df_sensitivity.pivot(index="Ridge Alpha", columns="Input Scaling", values="RMSE")

plt.subplot(1, 2, 2)
sns.heatmap(pivot_sensitivity, annot=True, fmt=".2f", cmap="RdYlGn_r", cbar_kws={'label': 'RMSE'})
plt.title("Hyperparameter Sensitivity\n(Fixed: Window=48, Layers=1, Hidden=50)", pad=20)

plt.tight_layout()
# FIX: bbox_inches='tight' prevents title/label clipping
plt.savefig('rvfl_heatmaps_fixed.png', dpi=300, bbox_inches='tight')
plt.show()


# 3. RADAR CHART (FIXED)
categories = ['Window', 'Hidden Size', 'Num Layers', 'Ridge Alpha', 'Input Scaling']

def normalize(row):
    # Mapping values to a 0-1 scale for visualization
    vals = []
    vals.append((row['Window'] - 48) / (124 - 48))
    vals.append((row['Hidden Size'] - 20) / (200 - 20))
    vals.append((row['Num Layers'] - 1) / (7 - 1))
    vals.append((np.log10(row['Ridge Alpha']) - np.log10(0.001)) / (np.log10(1.0) - np.log10(0.001)))
    vals.append((row['Input Scaling'] - 0.1) / (1.0 - 0.1))
    return vals

mean_coords = normalize(top_mean)
median_coords = normalize(top_median)
mean_coords += mean_coords[:1]
median_coords += median_coords[:1]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# FIX: Add internal padding for polar labels
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)

ax.plot(angles, mean_coords, color='#1f77b4', linewidth=2, label='Top Mean Config')
ax.fill(angles, mean_coords, color='#1f77b4', alpha=0.25)
ax.plot(angles, median_coords, color='#ff7f0e', linewidth=2, linestyle='--', label='Top Median Config')
ax.fill(angles, median_coords, color='#ff7f0e', alpha=0.25)

ax.set_thetagrids(np.degrees(angles[:-1]), categories)

# FIX: Adjust label alignment so they don't touch the circle
for label, angle in zip(ax.get_xticklabels(), angles):
    if 0 < angle < np.pi:
        label.set_horizontalalignment('left')
    elif angle > np.pi:
        label.set_horizontalalignment('right')

plt.title("Hyperparameter Signature: Mean vs Median", y=1.1, fontsize=14)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# FIX: bbox_inches='tight' is essential for legends outside the plot
plt.savefig('rvfl_radar_comparison_fixed.png', dpi=300, bbox_inches='tight')
plt.show()