"""Plot the recorded case-level measurements and write metric summaries."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASETS = [
    ("RIRE_CT_MRI", "RIRE"),
    ("AbdomenMRCT_CT_MRI", "AbdomenMRCT"),
    ("LungCT_CT_CT", "LungCT"),
]
METHODS = [("python", "FOTMIR", "#237A9B"), ("icp", "ICP", "#E39C39"),
           ("fgr", "FGR", "#56A274"), ("ants", "ANTs", "#B16B96")]
METRICS = [
    ("hd95", "primary_hd95_mm", "HD95", "mm", "lower"),
    ("asd", "primary_asd_mm", "ASD", "mm", "lower"),
    ("assd", "primary_assd_mm", "ASSD", "mm", "lower"),
    ("rmse", "rmse_mm", "Point-set RMSE", "mm", "lower"),
    ("runtime", "elapsed_time_s", "Registration runtime", "s", "lower"),
    ("dice", "dice", "Organ-union Dice", "unitless", "higher"),
    ("mutual_information", "mi", "Organ-region MI", "nats", "higher"),
    ("tre_mean", "tre_mean_mm", "Mean TRE", "mm", "lower"),
    ("tre95", "tre95_mm", "TRE95", "mm", "lower"),
]


def main():
    data = pd.read_csv(ROOT / "raw/case_metrics.csv")
    figures = ROOT / "figures"
    summaries = ROOT / "summary"
    figures.mkdir(exist_ok=True)
    summaries.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "axes.titlesize": 18, "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9AA2A9", "text.color": "#243340",
        "axes.labelcolor": "#243340", "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    rows = []
    for stem, field, title, unit, preferred in METRICS:
        values = pd.to_numeric(data[field], errors="coerce")
        finite = np.isfinite(values)
        active = [(key, label) for key, label in DATASETS
                  if (finite & (data.dataset == key)).any()]
        width = 7.8 if len(active) == 1 else 5.7 * len(active) + 0.6
        fig, axes = plt.subplots(1, len(active), squeeze=False,
                                 figsize=(width, 6.4))
        for ax, (dataset, display) in zip(axes[0], active):
            panel_max = 0.0
            for x, (method, label, color) in enumerate(METHODS):
                series = values[finite & (data.dataset == dataset) & (data.method == method)]
                if series.empty:
                    continue
                mean, median = float(series.mean()), float(series.median())
                low, high = float(series.min()), float(series.max())
                panel_max = max(panel_max, high)
                rows.append(dict(dataset=dataset, method=method, metric=stem,
                                 source_column=field, unit=unit, count=len(series),
                                 mean=mean, median=median, minimum=low, maximum=high))
                ax.bar(x, mean, width=0.64, color=color, alpha=0.93, zorder=3)
                ax.errorbar(x, mean, yerr=[[max(0, mean-low)], [max(0, high-mean)]],
                            fmt="none", ecolor="#303C46", elinewidth=1.9,
                            capsize=7, capthick=1.9, zorder=4)
                ax.plot([x-.20, x+.20], [median, median], color="white", lw=2.6,
                        path_effects=[pe.Stroke(linewidth=4.2, foreground="#303C46"),
                                      pe.Normal()], zorder=5)
            ax.set_title(display, fontweight="bold", pad=14)
            ax.set_xticks(range(len(METHODS)), [x[1] for x in METHODS])
            ax.set_xlim(-0.65, len(METHODS)-0.35)
            ax.set_ylim(0, max(panel_max * 1.10, 0.01))
            ax.set_ylabel(f"{title} ({unit})" if unit != "unitless" else title)
            ax.grid(axis="y", color="#DFE5E9", linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", length=0, pad=8)
        fig.suptitle(f"{title}  |  {preferred.capitalize()} is better", fontsize=22,
                     fontweight="bold", y=0.98)
        fig.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in METHODS],
                   loc="upper center", bbox_to_anchor=(0.5, 0.91),
                   ncol=4, frameon=False, fontsize=14, handlelength=1.4,
                   columnspacing=1.5)
        fig.subplots_adjust(top=0.76, bottom=0.12, left=0.15 if len(active)==1 else 0.06,
                            right=0.97, wspace=0.34)
        fig.savefig(figures / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.20)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(summaries / "metric_summary.csv", index=False)
    print(f"Created {len(METRICS)} metric images and metric_summary.csv")


if __name__ == "__main__":
    main()
