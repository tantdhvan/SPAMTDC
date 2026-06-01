import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


METRICS = [
    ("Objective value", "f_value"),
    ("Consistency violations", "consistency_violations"),
    ("Value-oracle queries", "queries"),
    ("Running time (sec)", "time_sec"),
]

ROUTE_ORDER = ["OPS", "GongD1", "Pham22"]
MARKERS = {"OPS": "o", "GongD1": "s", "Pham22": "^"}
COLORS = {"OPS": "tab:blue", "GongD1": "tab:orange", "Pham22": "tab:green"}
# Legend display names: the executable/CSV route label "OPS" is the legacy
# identifier for the algorithm presented as STAMP in the manuscript.
DISPLAY = {"OPS": "STAMP"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build active OPS/GongD1/Pham22 metric charts from a CSV trace."
    )
    parser.add_argument("input_csv", help="CSV produced by the experiment driver")
    parser.add_argument("--output", default=None, help="Output PDF path")
    parser.add_argument("--title", default=None, help="Dataset title override")
    return parser.parse_args()


def route_order(routes):
    preferred = [route for route in ROUTE_ORDER if route in routes]
    rest = sorted(route for route in routes if route not in ROUTE_ORDER)
    return preferred + rest


def dataset_title(df, override):
    if override:
        return override
    if df.empty:
        return "Active metrics"
    row = df.iloc[0]
    graph = row.get("graph", "dataset")
    n = row.get("n", "?")
    k = row.get("K", "?")
    b = row.get("B", "?")
    return f"{graph} (n={n}, K={k}, B={b})"


def aggregate(df, metric):
    return (
        df[["route", "t", metric]]
        .dropna()
        .groupby(["route", "t"], as_index=False)[metric]
        .mean()
    )


def main():
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".metrics.pdf")

    df = pd.read_csv(input_path)
    missing = {"route", "t"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    routes = route_order(df["route"].dropna().unique().tolist())
    title = dataset_title(df, args.title)

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 300,
    })

    with PdfPages(output_path) as pdf:
        for label, metric in METRICS:
            if metric not in df.columns:
                continue
            agg = aggregate(df, metric)
            fig, ax = plt.subplots(figsize=(7.0, 4.4))

            for route in routes:
                sub = agg[agg["route"] == route].sort_values("t")
                if sub.empty:
                    continue
                ax.plot(
                    sub["t"],
                    sub[metric],
                    label=DISPLAY.get(route, route),
                    marker=MARKERS.get(route, "o"),
                    color=COLORS.get(route, None),
                    linewidth=1.2,
                    markersize=4,
                )

            ax.set_title(title)
            ax.set_xlabel("Prefix t")
            ax.set_ylabel(label)
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
            ax.legend(frameon=False)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Created {output_path}")


if __name__ == "__main__":
    main()
