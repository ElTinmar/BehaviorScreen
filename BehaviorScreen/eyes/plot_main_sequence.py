"""
plot_main_sequence.py
======================
Velocity vs. amplitude "main sequence" plot per cluster, with a
saturating-exponential fit (V = Vmax * (1 - exp(-amp/k))), matching the
model used throughout BF2_ExpFits*.m. A visible saturating relationship
is a strong scientific validity check for genuine saccades.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def sat_exp(amp, vmax, k):
    return vmax * (1 - np.exp(-amp / k))


def plot_main_sequence(csv_path: Path, output_png: Path | None = None) -> None:
    events = pd.read_csv(csv_path)

    # combine L/R: use whichever eye's amplitude is larger in magnitude
    # as "the" amplitude for that event, paired with that eye's cw/ccw
    # velocity (whichever matches the direction of the amplitude change)
    amp_L, amp_R = events["Amp_L"].to_numpy(), events["Amp_R"].to_numpy()
    vel_cw_L, vel_ccw_L = events["Vel_cw_L"].to_numpy(), events["Vel_ccw_L"].to_numpy()
    vel_cw_R, vel_ccw_R = events["Vel_cw_R"].to_numpy(), events["Vel_ccw_R"].to_numpy()

    use_L = np.abs(amp_L) >= np.abs(amp_R)
    amp = np.where(use_L, amp_L, amp_R)
    vel = np.where(
        use_L,
        np.where(amp_L >= 0, vel_cw_L, vel_ccw_L),
        np.where(amp_R >= 0, vel_cw_R, vel_ccw_R),
    )

    clusters = sorted(c for c in events["cluster"].unique() if c != -1)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")

    for i, cl in enumerate(clusters):
        mask = (events["cluster"] == cl).to_numpy()
        x, y = np.abs(amp[mask]), np.abs(vel[mask])
        valid = (x > 0) & (y > 0) & (y < 1500) & ~np.isnan(x) & ~np.isnan(y)
        x, y = x[valid], y[valid]
        if len(x) < 10:
            continue

        ax.scatter(x, y, s=8, alpha=0.15, color=cmap(i % 10), rasterized=True)

        try:
            popt, _ = curve_fit(sat_exp, x, y, p0=[900, 5], bounds=([0, 0], [5000, 1000]))
            x_fit = np.linspace(0, x.max(), 100)
            ax.plot(x_fit, sat_exp(x_fit, *popt), color=cmap(i % 10), lw=2,
                    label=f"cluster {cl}: Vmax={popt[0]:.0f}, k={popt[1]:.1f}")
        except RuntimeError:
            print(f"[warn] main sequence fit failed for cluster {cl}")

    ax.set_xlabel("Saccade amplitude (deg)")
    ax.set_ylabel("Peak eye velocity (deg/s)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()

    if output_png is not None:
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    plot_main_sequence(args.csv, args.output)