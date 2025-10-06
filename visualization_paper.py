import pickle
import utils.datacleaner as dc
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


def analyze_success_from_pkl_with_process_results(pkl_path: str, threshold: float, test_name: str) -> None:
    """
    Carica un file .pkl con 'stats_per_points', lo passa a process_results(stats_per_points)
    per ottenere un DataFrame con colonne: ['Method', 'Regressor Points', 'Resampling Points', 'Epsilons'].
    Quindi:
      - stampa conteggio globale di valori Epsilons > threshold
      - calcola e stampa il success-rate per Method (percentuale di Epsilons > threshold)
      - salva un barplot + tabella (PNG) e un CSV con i risultati in visualization/figs/figs_{test_name}/
    """

    out_dir = f"visualization/figs/figs_{test_name}"
    os.makedirs(out_dir, exist_ok=True)

    with open(pkl_path, "rb") as f:
        stats_per_points = pickle.load(f)

    df = dc.process_results(stats_per_points)

    # --- success-rate per Method ---
    grouped = (
        df.assign(above=lambda d: d["Epsilons"] >= threshold)
          .groupby("Method", as_index=False)
          .agg(above_count=("above", "sum"), total=("Epsilons", "count"))
    )

    grouped["success_rate_%"] = grouped["above_count"] / grouped["total"] * 100.0
    
    grouped = grouped.sort_values("success_rate_%", ascending=False).reset_index(drop=True)

    print("Success rate per 'Method':")
    for _, row in grouped.iterrows():
        print(f"- {row['Method']}: {int(row['above_count'])}/{int(row['total'])} = {row['success_rate_%']:.2f}%")

    # --- salva CSV ---
    csv_path = os.path.join(out_dir, f"success_rates_threshold_{str(threshold).replace('.', '_')}.csv")
    grouped.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")

    # --- figura: barplot + tabella (stile simile al tuo esempio) ---
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    # Subplot 1: barplot
    ax_plot = plt.subplot(gs[0])
    sns.barplot(data=grouped, x="Method", y="success_rate_%", ax=ax_plot)
    ax_plot.set_title(f"Success Rate per Method — threshold = {threshold}")
    ax_plot.set_xlabel("Method")
    ax_plot.set_ylabel("Success rate sopra soglia (%)")
    ax_plot.set_xticklabels(ax_plot.get_xticklabels(), rotation=30, ha="right")
    ax_plot.grid(True, axis='y', linestyle=':', alpha=0.7)

    # Annotazioni barre
    for i, p in enumerate(ax_plot.patches):
        rate = grouped.loc[i, "success_rate_%"]
        a = int(grouped.loc[i, "above_count"])
        t = int(grouped.loc[i, "total"])
        ax_plot.text(
            p.get_x() + p.get_width()/2,
            p.get_height(),
            f"{rate:.1f}%\n({a}/{t})",
            ha="center", va="bottom", fontsize=9
        )

    # Subplot 2: tabella
    display_cols = ["Method", "above_count", "total", "success_rate_%"]
    df_table = grouped[display_cols].copy()
    df_table.rename(columns={
        "Method": "Method",
        "above_count": "Above",
        "total": "Total",
        "success_rate_%": "Success Rate (%)"
    }, inplace=True)

    ax_table = plt.subplot(gs[1])
    ax_table.axis('off')
    table = ax_table.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.4)

    # save
    img_path = os.path.join(out_dir, f"success_rates_threshold_{str(threshold).replace('.', '_')}.png")
    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    plt.close()

    print(f"Saved bar plot: {img_path}")

if __name__ == "__main__":
    pkl_path = 'visualization/data/data_regressor_500_over_20_amd_40_regr_gt/oversampling_results_invalid_configs.pkl'
    threshold = 0.5
    test_name = "test_1"

    analyze_success_from_pkl_with_process_results(pkl_path, 0.6, "test_1")