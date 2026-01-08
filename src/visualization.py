import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_metric_per_attribute(report_path, metric='F1', stage='Base', output_dir='plots'):
    """
    Generates a horizontal bar chart for a specific metric (e.g., Base_F1).
    """
    if not os.path.exists(report_path):
        print(f"Error: Report file not found at {report_path}")
        return

    # 1. Load Data
    df = pd.read_csv(report_path)
    
    # Construct column name, e.g., 'Base_F1'
    col_name = f"{stage}_{metric}"
    
    if col_name not in df.columns:
        print(f"Error: Column {col_name} not found in report.")
        return

    # 2. Sort for readability (Best to Worst)
    df_sorted = df.sort_values(by=col_name, ascending=True)

    # 3. Plotting
    plt.figure(figsize=(10, 12)) # Tall plot for 40 attributes
    sns.set_style("whitegrid")
    
    # Color mapping: darker blue for higher values
    norm = plt.Normalize(df_sorted[col_name].min(), df_sorted[col_name].max())
    colors = plt.cm.viridis(norm(df_sorted[col_name].values))
    
    bars = plt.barh(df_sorted['Attribute'], df_sorted[col_name], color=colors)
    
    # Add vertical line for the Mean
    mean_val = df_sorted[col_name].mean()
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean {metric}: {mean_val:.3f}')
    
    # Aesthetics
    plt.title(f"{stage} {metric} Score per Attribute (Untrained Head)", fontsize=15)
    plt.xlabel(f"{metric} Score", fontsize=12)
    plt.ylabel("Attribute", fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    # 4. Save
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{stage.lower()}_{metric.lower()}_distribution.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.show()

def plot_comparison(report_path, output_dir='plots'):
    """
    (Future Proofing)
    If the report has Attack/Defense columns, plot grouped bars to compare them.
    Currently just a placeholder or could plot Base vs Random chance.
    """
    pass

if __name__ == "__main__":
    # Quick standalone test
    plot_metric_per_attribute("reports/model_comparison.csv", metric='F1', stage='Base')