#!/usr/bin/env python3
import glob
import pandas as pd


def main():
    files = sorted(glob.glob("results/metrics_*_line_none.csv"))

    if not files:
        print("No metrics files found. Run ./run_experiments.sh first.")
        return

    rows = []
    for path in files:
        rows.append(pd.read_csv(path))

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(["xy_rmse", "resets"])

    print("\n=== Controller Comparison ===")
    print(df.to_string(index=False))

    df.to_csv("results/summary_metrics.csv", index=False)

    with open("results/summary_metrics.md", "w") as f:
        f.write("# Controller Comparison Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print("\nSaved:")
    print("  results/summary_metrics.csv")
    print("  results/summary_metrics.md")


if __name__ == "__main__":
    main()
