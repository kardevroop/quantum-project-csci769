import csv
import os
import matplotlib.pyplot as plt
from objects import ExperimentSummary

def summarize_by_depth(summary: ExperimentSummary) -> list[dict]:
    """Aggregate experiment results into table-friendly rows.

    Each row corresponds to one (setting, depth) combination and records
    the metrics required in the proposal.

    Args:
        summary: Completed experiment summary.

    Returns:
        A list of dictionaries suitable for CSV export and plotting.
    """
    rows: list[dict] = []

    for r in summary.results:
        rows.append(
            {
                "setting": r.setting,
                "depth": r.depth,
                "best_bitstring": r.best_bitstring,
                "cut_value": r.best_cut_value,
                "exact_cut_value": r.exact_cut_value,
                "approximation_ratio": r.approximation_ratio,
                "runtime_seconds": r.runtime_seconds,
                "shots": r.shots,
                "optimizer_success": r.optimizer_success,
            }
        )

    rows.sort(key=lambda x: (x["setting"], x["depth"]))
    return rows


def save_results_csv(summary: ExperimentSummary, filename: str = "qaoa_results.csv") -> None:
    """Save experiment metrics to a CSV file.

    Args:
        summary: Completed experiment summary.
        filename: Output CSV path.
    """
    rows = summarize_by_depth(summary)
    if not rows:
        print("No results available to save.")
        return

    fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results CSV to: {filename}")


def group_rows_by_setting(rows: list[dict]) -> dict[str, list[dict]]:
    """Group result rows by execution setting.

    Args:
        rows: Flat list of result dictionaries.

    Returns:
        Mapping from setting name to sorted rows.
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["setting"], []).append(row)

    for setting in grouped:
        grouped[setting].sort(key=lambda x: x["depth"])

    return grouped


def plot_metric_vs_depth(
    rows: list[dict],
    metric_key: str,
    ylabel: str,
    filename: str,
    title: str,
) -> None:
    """Plot a metric as a function of QAOA depth.

    A separate line is drawn for each execution setting.

    Args:
        rows: Table of result rows.
        metric_key: Dictionary key for the metric to plot.
        ylabel: Label for the y-axis.
        filename: Output image filename.
        title: Plot title.
    """
    grouped = group_rows_by_setting(rows)

    plt.figure(figsize=(8, 5))

    for setting, setting_rows in grouped.items():
        depths = [row["depth"] for row in setting_rows]
        values = [row[metric_key] for row in setting_rows]
        plt.plot(depths, values, marker="o", label=setting)

    plt.xlabel("QAOA depth (p)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted({row["depth"] for row in rows}))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved plot to: {filename}")


def generate_analysis_plots(summary: ExperimentSummary, outdir: str = "results") -> None:
    """Generate all proposal-motivated analysis plots.

    Creates:
        - cut value vs depth
        - approximation ratio vs depth
        - runtime vs depth

    Args:
        summary: Completed experiment summary.
        outdir: Output directory for plots.
    """
    os.makedirs(outdir, exist_ok=True)
    rows = summarize_by_depth(summary)

    if not rows:
        print("No results available for plotting.")
        return

    plot_metric_vs_depth(
        rows=rows,
        metric_key="cut_value",
        ylabel="Cut value",
        filename=os.path.join(outdir, "cut_value_vs_depth.png"),
        title="QAOA Cut Value vs Circuit Depth",
    )

    plot_metric_vs_depth(
        rows=rows,
        metric_key="approximation_ratio",
        ylabel="Approximation ratio",
        filename=os.path.join(outdir, "approximation_ratio_vs_depth.png"),
        title="QAOA Approximation Ratio vs Circuit Depth",
    )

    plot_metric_vs_depth(
        rows=rows,
        metric_key="runtime_seconds",
        ylabel="Runtime (seconds)",
        filename=os.path.join(outdir, "runtime_vs_depth.png"),
        title="QAOA Runtime vs Circuit Depth",
    )


def print_depth_analysis(summary: ExperimentSummary) -> None:
    """Print a concise analysis of how depth affects solution quality.

    Args:
        summary: Completed experiment summary.
    """
    rows = summarize_by_depth(summary)
    grouped = group_rows_by_setting(rows)

    print("\n" + "=" * 72)
    print("DEPTH ANALYSIS")
    print("=" * 72)

    for setting, setting_rows in grouped.items():
        print(f"\nSetting: {setting}")

        best_row = max(setting_rows, key=lambda x: x["approximation_ratio"])
        fastest_row = min(setting_rows, key=lambda x: x["runtime_seconds"])

        print("Depth-by-depth metrics:")
        for row in setting_rows:
            print(
                f"  p={row['depth']}: "
                f"cut={row['cut_value']}, "
                f"ratio={row['approximation_ratio']:.4f}, "
                f"runtime={row['runtime_seconds']:.4f}s"
            )

        print(
            f"  Best solution quality at p={best_row['depth']} "
            f"(ratio={best_row['approximation_ratio']:.4f}, "
            f"cut={best_row['cut_value']})"
        )
        print(
            f"  Fastest execution at p={fastest_row['depth']} "
            f"(runtime={fastest_row['runtime_seconds']:.4f}s)"
        )

        if len(setting_rows) >= 2:
            first = setting_rows[0]
            last = setting_rows[-1]
            print(
                f"  Change from p={first['depth']} to p={last['depth']}: "
                f"cut {first['cut_value']} -> {last['cut_value']}, "
                f"ratio {first['approximation_ratio']:.4f} -> {last['approximation_ratio']:.4f}, "
                f"runtime {first['runtime_seconds']:.4f}s -> {last['runtime_seconds']:.4f}s"
            )

    print("=" * 72)


def generate_full_report(summary: ExperimentSummary, outdir: str = "results") -> None:
    """Generate all tables, CSV files, plots, and printed analysis.

    Args:
        summary: Completed experiment summary.
        outdir: Directory for saved outputs.
    """
    os.makedirs(outdir, exist_ok=True)

    csv_path = os.path.join(outdir, "qaoa_results.csv")
    save_results_csv(summary, filename=csv_path)
    generate_analysis_plots(summary, outdir=outdir)
    print_depth_analysis(summary)