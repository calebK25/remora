# Technical study script 

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    #Load the Excel file and tidy column types/ordering.
    df = pd.read_excel(path, parse_dates=["date"], dtype={"security": str})
    df = df.dropna(subset=["security", "PX_LAST"])
    df["security"] = df["security"].str.strip()  # remove stray spaces / newlines
    df = df.sort_values(["security", "date"]).reset_index(drop=True)
    return df


def compute_event_metrics(
    df: pd.DataFrame, windows: List[int], threshold: float, direction: str
) -> pd.DataFrame:

    # Pre‑compute rolling returns for every requested window
    for k in windows:
        df[f"ret_{k}"] = (
            df.groupby("security")["PX_LAST"].transform(lambda s: s.pct_change(k))
        )

    summaries = []

    # Loop over each window size and each security
    for k in windows:
        fw_returns: list[float] = []

        for _, grp in df.groupby("security", sort=False):
            grp = grp.reset_index(drop=True)

            event_mask = (
                grp[f"ret_{k}"] >= threshold
                if direction == "up"
                else grp[f"ret_{k}"] <= -threshold
            )

            event_indices = np.where(event_mask)[0]
            for idx in event_indices:
                target_idx = idx + k
                if target_idx >= len(grp):
                    continue  # insufficient forward data

                # Row‑level lookup via iloc to avoid any mis‑alignment issues**
                sec_name = grp.iloc[idx]["security"]
                start_price = grp.iloc[idx]["PX_LAST"]
                end_price = grp.iloc[target_idx]["PX_LAST"]

                if pd.notna(start_price) and pd.notna(end_price):
                    fw_returns.append(end_price / start_price - 1)

        summaries.append(
            {
                "window": k,
                "n_events": len(fw_returns),
                "avg_change": np.mean(fw_returns) if fw_returns else np.nan,
                "pct_gainers": np.mean(np.array(fw_returns) > 0) if fw_returns else np.nan,
            }
        )

    return pd.DataFrame(summaries)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Technical reversion/momentum event study")
    p.add_argument("--file", default="Technical_Study.xlsx", help="Path to Excel file")
    p.add_argument(
        "--windows",
        "-w",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Input / prediction window lengths in trading days",
    )
    p.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.05,
        help="Percentage threshold that defines an event (e.g. 0.05 = 5%)",
    )
    p.add_argument(
        "--direction",
        "-d",
        choices=["up", "down"],
        default="down",
        help="Look for up‑moves or down‑moves",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(Path(args.file))
    summary = compute_event_metrics(
        df, windows=args.windows, threshold=args.threshold, direction=args.direction
    )

    print("\nEvent Study Summary\n" + "-" * 22)
    print(summary.to_string(index=False, float_format="{:.4f}".format))

    out_path = Path("event_study_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path.resolve()}")


if __name__ == "__main__":
    main()
