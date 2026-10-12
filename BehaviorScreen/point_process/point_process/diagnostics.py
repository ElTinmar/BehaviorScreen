"""Diagnostic dashboard and model plots."""
from __future__ import annotations

import string
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .dataset import PointProcessDataset

if TYPE_CHECKING:
    from .point_process import PointProcess

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
OBSERVED = SERIES[0]
BAND = "#9a9994"
INK = "#52514e"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_orange", [SERIES[0], "#f0efec", SERIES[1]])
CHECK_LABELS = {
    "calibration_ks": "Calibration KS",
    "calibration_cvm": "Calibration CvM",
    "max_abs_event_acf": "Max |interval ACF|",

    "zero_stream_fraction": "Trials without event",
    "multiple_event_stream_fraction": "Trials with >= 2 events",
    "mean_stream_count": "Mean events / trial",
    "variance_stream_count": "Var events / trial",
    "variance_fish_total": "Var events / fish",
    "max_fish_total": "Max events / fish",
    "gap_q10": "Post-event gap, 10th pct (s)",
    "gap_median": "Post-event gap, median (s)",
}


def _percentiles(curves: Sequence[np.ndarray], ci: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise lower, median and upper percentiles across replicates (NaN where none)."""
    alpha = (100.0 - ci) / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return tuple(np.nanpercentile(np.asarray(curves, dtype=float), [alpha, 50.0, 100.0 - alpha], axis=0))


def _p_value(gof: Dict[str, Any], statistic: str) -> str:
    summary = gof["summary"]
    rows = summary[summary["statistic"] == statistic] if len(summary) else summary
    return f"p={rows['p_value'].iloc[0]:.3f}" if len(rows) else "p n/a"


def _style(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


class DiagnosticsMixin:
    def plot_panel_calibration(self, gof: Dict[str, Any], ax: plt.Axes) -> None:
        """KS plot of the rescaled intervals (KM-based), with the bootstrap envelope and band."""
        u = gof["u_grid"]
        if gof["replicates"]:
            ax.fill_between(u, gof["simultaneous_cdf_lower"], gof["simultaneous_cdf_upper"], color=BAND, alpha=0.2, lw=0,
                            label=f"Simultaneous {gof['ci']:.0f}% envelope")
            ax.fill_between(u, gof["bootstrap_cdf_lower"], gof["bootstrap_cdf_upper"], color=BAND, alpha=0.35, lw=0,
                            label=f"Pointwise {gof['ci']:.0f}% band")
        ax.plot([0, 1], [0, 1], "--", color=INK, lw=1, label="Uniform")
        ax.step(u, gof["observed"]["calibration_cdf"], where="post", color=OBSERVED, lw=2, label="Observed")
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_title(f"Timing calibration\nKS {_p_value(gof, 'calibration_ks')}, CvM {_p_value(gof, 'calibration_cvm')}")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        _style(ax, "Rescaled interval, uniform scale $u$", "Kaplan-Meier CDF")

    def plot_panel_localization(self, gof: Dict[str, Any], grouping: str, title: str, ax: plt.Axes) -> None:
        """Cox-Snell curves of one grouping of the intervals, with pointwise bootstrap bands."""
        r = gof["r_grid"]
        curves = gof["observed"]["localization"][grouping]
        for color, (group, curve) in zip(SERIES, curves.items()):
            if gof["replicates"]:
                lower, _, upper = _percentiles([rep["localization"][grouping][group] for rep in gof["replicates"]], gof["ci"])
                ax.fill_between(r, lower, upper, color=color, alpha=0.15, lw=0)
            if np.isfinite(curve).any():
                ax.plot(r, curve, color=color, lw=1.8, label=group)
        finite = [c[np.isfinite(c)] for c in curves.values() if np.isfinite(c).any()]
        top = max([c.max() for c in finite] + [1.0]) * 1.05
        ax.plot([0, top], [0, top], "--", color=INK, lw=1, label="Model")
        ax.set(xlim=(0, top), ylim=(0, top))
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        _style(ax, "Cox-Snell residual $r$", r"$-\log \hat S_{KM}(r)$")

    def plot_panel_post_event_gaps(self, gof: Dict[str, Any], ax: plt.Axes) -> None:
        """Density of the time from one event to the next within a trial, on a log time axis."""
        edges = gof["gap_edges"]
        density = lambda counts: counts / max(counts.sum(), 1) / np.diff(np.log10(edges))
        if gof["replicates"]:
            lower, median, upper = _percentiles([density(rep["gap_counts"]) for rep in gof["replicates"]], gof["ci"])
            ax.fill_between(edges[:-1], lower, upper, step="post", color=BAND, alpha=0.35, lw=0,
                            label=f"Bootstrap {gof['ci']:.0f}% band")
            ax.stairs(median, edges, color=INK, lw=1, ls="--", label="Bootstrap median")
        ax.stairs(density(gof["observed"]["gap_counts"]), edges, color=OBSERVED, lw=2, label="Observed")
        ax.set_xscale("log")
        ax.set_title(f"Gap after an event\n10th percentile {_p_value(gof, 'gap_q10')}, median {_p_value(gof, 'gap_median')}")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        _style(ax, "Time since previous event (s)", "Density per log10 s")

    def plot_panel_interval_acf(self, gof: Dict[str, Any], ax: plt.Axes) -> None:
        """Autocorrelation of successive rescaled intervals within trials."""
        observed = gof["observed"]
        lags = np.arange(1, len(observed["acf"]) + 1)
        if gof["replicates"]:
            lower, _, upper = _percentiles([rep["acf"] for rep in gof["replicates"]], gof["ci"])
            label = f"Bootstrap {gof['ci']:.0f}% band"
        else:
            with np.errstate(divide="ignore"):
                upper = 1.96 / np.sqrt(observed["acf_pairs"])
            lower, label = -upper, "±1.96/sqrt(pairs)"
        ax.fill_between(lags, lower, upper, color=BAND, alpha=0.35, lw=0, step="mid", label=label)
        ax.vlines(lags, 0, observed["acf"], color=OBSERVED, lw=2, label="Observed")
        ax.axhline(0, color=INK, lw=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"Interval autocorrelation\nmax |ACF| {_p_value(gof, 'max_abs_event_acf')}")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        _style(ax, "Lag (events)", "Autocorrelation")

    def plot_panel_first_event_survival(self, dataset: PointProcessDataset, gof: Dict[str, Any], ax: plt.Axes) -> None:
        """Kaplan-Meier first-event survival against replicates (or the model curve without them)."""
        t = gof["survival_grid"]
        if gof["replicates"]:
            lower, median, upper = _percentiles([rep["first_event_survival"] for rep in gof["replicates"]], gof["ci"])
            ax.fill_between(t, lower, upper, color=BAND, alpha=0.35, lw=0, step="post", label=f"Bootstrap {gof['ci']:.0f}% band")
            ax.step(t, median, where="post", color=INK, lw=1, ls="--", label="Bootstrap median")
        else:
            try:
                model_t, model_survival = self.population_survival_curve(dataset)
                ax.plot(model_t, model_survival, "--", color=INK, lw=1, label="Model")
            except NotImplementedError:
                pass
        ax.step(t, gof["observed"]["first_event_survival"], where="post", color=OBSERVED, lw=2, label="Observed")
        ax.set_ylim(0, 1.02)
        ax.set_title("First-event survival")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        _style(ax, "Time in trial (s)", "P(no event yet)")

    def plot_panel_residual_surface(
        self, dataset: PointProcessDataset, residuals: Dict[str, np.ndarray], ax: plt.Axes, n_time_bins: int = 48
    ) -> None:
        """
        Pearson residuals of the counts over trial x time in trial, in about
        n_time_bins bins (variance with the per-fish gain shared across bins).
        """
        width = max(1, residuals["y_obs"].shape[1] // n_time_bins)
        coarse = lambda x: np.add.reduceat(x, np.arange(0, x.shape[1], width), axis=1)
        observed, expected = coarse(residuals["y_obs"]), coarse(residuals["mu_pred"])
        r_eff = residuals["r_eff"]
        with np.errstate(divide="ignore", invalid="ignore"):
            pearson = (observed - expected) / np.sqrt(expected + expected**2 / r_eff)
        vmax = np.nanpercentile(np.abs(pearson), 98) if np.isfinite(pearson).any() else 1.0
        image = ax.imshow(
            pearson, aspect="auto", origin="lower", cmap=DIVERGING, vmin=-vmax, vmax=vmax, interpolation="nearest",
            extent=[0.0, dataset.duration_s, -0.5, dataset.num_trials - 0.5],
        )
        ax.set_title(f"Count residuals ({width * dataset.binning_dt:g} s bins)")
        ax.set_xlabel("Time in trial (s)")
        ax.set_ylabel("Trial (presentation order)")
        ax.figure.colorbar(image, ax=ax, label="Pearson residual")

    def plot_panel_stream_counts(self, gof: Dict[str, Any], ax: plt.Axes) -> None:
        """Distribution of events per trial: observed against replicates."""
        observed = gof["observed"]["stream_counts"]
        k_max = int(np.clip(np.percentile(observed, 99), 3, 15))
        proportions = lambda counts: np.bincount(np.minimum(counts, k_max), minlength=k_max + 1) / len(counts)
        k = np.arange(k_max + 1)
        ax.bar(k, proportions(observed), width=0.6, color=OBSERVED, label="Observed")
        if gof["replicates"]:
            lower, median, upper = _percentiles([proportions(rep["stream_counts"]) for rep in gof["replicates"]], gof["ci"])
            ax.errorbar(k, median, yerr=[median - lower, upper - median], fmt="o", color=INK, ms=4, capsize=3,
                        label=f"Bootstrap median, {gof['ci']:.0f}%")
        ax.set_xticks(k, [str(i) for i in k[:-1]] + [f"{k_max}+"])
        ax.set_title(f"Events per trial\nno event {_p_value(gof, 'zero_stream_fraction')}, "
                     f"variance {_p_value(gof, 'variance_stream_count')}")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        _style(ax, "Events in trial", "Fraction of trials")

    def plot_panel_fish_totals(self, gof: Dict[str, Any], ax: plt.Axes) -> None:
        """Q-Q plot of events per fish: observed order statistics against replicates."""
        observed = gof["observed"]["fish_totals"]
        if not gof["replicates"]:
            ax.text(0.5, 0.5, "Needs bootstrap replicates", ha="center", va="center", transform=ax.transAxes)
        else:
            lower, median, upper = _percentiles([rep["fish_totals"] for rep in gof["replicates"]], gof["ci"])
            ax.fill_between(median, lower, upper, color=BAND, alpha=0.35, lw=0, label=f"Bootstrap {gof['ci']:.0f}% band")
            top = max(median.max(), observed.max()) * 1.05
            ax.plot([0, top], [0, top], "--", color=INK, lw=1, label="Model")
            ax.plot(median, observed, "o", color=OBSERVED, ms=3, label="Observed")
            ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.set_title(f"Events per fish (Q-Q)\nvariance {_p_value(gof, 'variance_fish_total')}, "
                     f"max {_p_value(gof, 'max_fish_total')}")
        _style(ax, "Simulated (bootstrap median order statistic)", "Observed")

    def plot_panel_checks(self, gof: Dict[str, Any], ax: plt.Axes) -> None:
        """Observed value, bootstrap interval and p-value of every check, and the omnibus p-value."""
        ax.axis("off")
        summary = gof["summary"]
        if not len(summary):
            ax.text(0.5, 0.5, "No bootstrap replicates", ha="center", va="center", transform=ax.transAxes)
            return
        rows = [
            [CHECK_LABELS[s], f"{o:.3g}", f"{m:.2g} [{lo:.2g}, {hi:.2g}]", f"{p:.3f}"]
            for s, o, m, lo, hi, p in summary[["statistic", "observed", "bootstrap_median", "ci_lower", "ci_upper", "p_value"]].itertuples(index=False)
        ]
        rows.append(["Omnibus (min p)", "", "", f"{gof['omnibus_p']:.3f}"])
        table = ax.table(cellText=rows, colLabels=["Check", "Observed", f"Bootstrap median [{gof['ci']:.0f}%]", "p"],
                         loc="center", cellLoc="left", colLoc="left", colWidths=[0.3, 0.15, 0.43, 0.12])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#d8d7d3")
            if row == 0 or row == len(rows):
                cell.set_text_props(fontweight="bold")
        ax.set_title(f"Checks against {len(gof['replicates'])} simulate-and-refit replicates")

    def plot_panel_parameter_correlation(self, correlation: np.ndarray, ax: plt.Axes) -> None:
        """Parameter correlation from the Hessian; parameters at a bound are marked."""
        image = ax.imshow(correlation, cmap=DIVERGING, vmin=-1.0, vmax=1.0)
        n = len(self.param_names)
        ax.set_xticks(np.arange(n), self.param_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(n), self.param_names, fontsize=8)
        for i in range(n):
            for j in range(n):
                value = correlation[i, j]
                ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "bound", ha="center", va="center", fontsize=7,
                        color="white" if np.isfinite(value) and abs(value) > 0.6 else "black")
        ax.set_title("Parameter correlation")
        ax.figure.colorbar(image, ax=ax, label="Correlation")

    def diagnose(
        self,
        dataset: PointProcessDataset,
        eps: float = 1e-5,
        run_parametric_gof: bool = True,
        gof_n_boot: int = 300,
        gof_seed: int = 42,
        gof_ci: float = 95.0,
        gof_acf_lags: int = 20,
        gof_min_km_at_risk: int = 1,
        gof_min_envelope_fraction: float = 0.80,
        gof_refit_n_starts: int = 1,
        gof_n_jobs: int = -1,
        gof_u_grid: Optional[np.ndarray] = None,
    ) -> Tuple[plt.Figure, Dict[str, Any]]:
        """
        Goodness-of-fit dashboard: every panel compares the data with
        simulate-and-refit replicates of the model (run_parametric_gof=False
        draws the observed curves only).
        """
        self._require_fitted()
        if run_parametric_gof:
            gof = self.parametric_gof_bootstrap(
                dataset, n_boot=gof_n_boot, seed=gof_seed, ci=gof_ci, acf_lags=gof_acf_lags,
                min_km_at_risk=gof_min_km_at_risk, min_envelope_fraction=gof_min_envelope_fraction,
                refit_n_starts=gof_refit_n_starts, n_jobs=gof_n_jobs, u_grid=gof_u_grid,
            )
        else:
            grids = self.gof_grids(dataset, gof_u_grid)
            observed = self.gof_statistics(dataset, **grids, acf_lags=gof_acf_lags, min_km_at_risk=gof_min_km_at_risk)
            gof = {**grids, "observed": observed, "replicates": [], "summary": pd.DataFrame(), "omnibus_p": np.nan, "ci": gof_ci}
        residuals = self.compute_residuals(dataset)
        correlation = self.estimate_parameter_correlation(dataset, eps=eps)

        panels: List[Callable[[plt.Axes], None]] = [
            lambda ax: self.plot_panel_calibration(gof, ax),
            lambda ax: self.plot_panel_checks(gof, ax),
            lambda ax: self.plot_panel_localization(gof, "trial_block", "Calibration by trial block", ax),
        ]
        if self.is_survival:
            panels.append(lambda ax: self.plot_panel_first_event_survival(dataset, gof, ax))
        else:
            panels += [
                lambda ax: self.plot_panel_localization(gof, "time_in_trial", "Calibration by interval start", ax),
                lambda ax: self.plot_panel_localization(gof, "event_order", "Calibration by event order", ax),
                lambda ax: self.plot_panel_post_event_gaps(gof, ax),
                lambda ax: self.plot_panel_interval_acf(gof, ax),
            ]
        panels += [
            lambda ax: self.plot_panel_residual_surface(dataset, residuals, ax),
            lambda ax: self.plot_panel_stream_counts(gof, ax),
            lambda ax: self.plot_panel_fish_totals(gof, ax),
            lambda ax: self.plot_panel_parameter_correlation(correlation, ax),
        ]

        n_rows = (len(panels) + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.6 * n_rows), layout="constrained")
        for letter, draw, ax in zip(string.ascii_uppercase, panels, axes.flat):
            draw(ax)
            title = ax.get_title()
            ax.set_title("")
            ax.set_title(f"{letter}. {title}", loc="left", fontsize=10, fontweight="bold")
        for ax in axes.flat[len(panels):]:
            ax.axis("off")
        fig.suptitle(self.latex_formula, fontsize=13)
        return fig, {"parametric_gof": gof, "residuals": residuals, "parameter_correlation": correlation}


class ModelPlotter:

    @staticmethod
    def _weighted_marginals(
        surface: np.ndarray, dataset: PointProcessDataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        time_marginal: rate(t), n_fish-weighted average across trials --
            same convention as dataset.time_histogram_hz, so empirical and
            model surfaces are reduced identically.
        trial_marginal: rate(trial), simple mean across time bins. Trials
            with n_fish_per_trial == 0 are NaN (not 0) -- distinguishes
            "no data" from "zero rate", matching
            plot_time_trial_rate_heatmap's convention.
        """
        n_fish = dataset.n_fish_per_trial
        active = n_fish > 0

        time_marginal = np.full(surface.shape[1], np.nan)
        if active.any():
            time_marginal = np.average(surface[active], axis=0, weights=n_fish[active])

        trial_marginal = np.full(surface.shape[0], np.nan)
        trial_marginal[active] = np.mean(surface[active], axis=1)

        return time_marginal, trial_marginal

    @staticmethod
    def plot_histogram(
        dataset: PointProcessDataset,
        model: PointProcess,
        figsize: Tuple[float, float] = (22, 8),
        cmap: str = "plasma",
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Empirical vs. model rate surface, each with overlaid empirical
        (solid) / model (dashed) marginals: one time-marginal strip
        spanning both heatmap columns, one shared trial-marginal column to
        the right. Same layout convention as the ablation_screen arm-
        surface-grid plots -- single shared GridSpec (not per-panel
        independent ones) so heatmaps and marginals stay pixel-aligned,
        and an explicit colorbar axis (not make_axes_locatable) so the
        colorbar never overlaps a title.
        """
        surf_emp = dataset.time_trial_histogram_hz
        surf_model = model.compute_expected_rate(dataset)

        tm_emp, trm_emp = ModelPlotter._weighted_marginals(surf_emp, dataset)
        tm_model, trm_model = ModelPlotter._weighted_marginals(surf_model, dataset)

        vmax = max(np.nanmax(surf_emp), np.nanmax(surf_model))
        vmax = max(vmax, 1e-9)
        time_ymax = max(
            np.nanmax(x) for x in [tm_emp, tm_model] if np.any(np.isfinite(x))
        )
        trial_xmax = max(
            np.nanmax(x) for x in [trm_emp, trm_model] if np.any(np.isfinite(x))
        )
        time_ymax, trial_xmax = max(time_ymax, 1e-9), max(trial_xmax, 1e-9)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            nrows=2,
            ncols=3,
            width_ratios=[4, 4, 1.3],
            height_ratios=[1, 4],
            wspace=0.12,
            hspace=0.12,
            left=0.07,
            right=0.86,
            top=0.90,
            bottom=0.09,
        )

        ax_time = fig.add_subplot(gs[0, 0:2])
        ax_emp = fig.add_subplot(gs[1, 0], sharex=ax_time)
        ax_mod = fig.add_subplot(gs[1, 1], sharex=ax_time, sharey=ax_emp)
        ax_trial = fig.add_subplot(gs[1, 2], sharey=ax_emp)

        ax_time.plot(
            dataset.t_centers,
            tm_emp,
            color="black",
            linestyle="-",
            linewidth=1.3,
            label="Empirical",
        )
        ax_time.plot(
            dataset.t_centers,
            tm_model,
            color="crimson",
            linestyle="--",
            linewidth=1.3,
            label="Model",
        )
        ax_time.set_ylim(0, time_ymax * 1.05)
        plt.setp(ax_time.get_xticklabels(), visible=False)
        ax_time.tick_params(labelsize=7)
        ax_time.legend(loc="upper right", fontsize=8, frameon=False)

        mesh = ax_emp.pcolormesh(
            dataset.t_grid,
            dataset.trial_edges,
            surf_emp,
            shading="flat",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )
        ax_mod.pcolormesh(
            dataset.t_grid,
            dataset.trial_edges,
            surf_model,
            shading="flat",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )
        ax_emp.set_title("Empirical Surface", fontsize=11, fontweight="bold")
        ax_mod.set_title(
            f"Fitted Surface: {model.name}", fontsize=11, fontweight="bold"
        )
        ax_emp.set_xlabel("Time in Trial (s)", fontsize=10)
        ax_mod.set_xlabel("Time in Trial (s)", fontsize=10)
        ax_emp.set_ylabel("Trial Number", fontsize=10)
        plt.setp(ax_mod.get_yticklabels(), visible=False)
        ax_emp.set_xlim(dataset.t_grid[0], dataset.t_grid[-1])
        ax_emp.tick_params(labelsize=8)
        ax_mod.tick_params(labelsize=8)

        trial_centers = np.arange(dataset.num_trials)
        ax_trial.plot(
            trm_emp, trial_centers, color="black", linestyle="-", linewidth=1.3
        )
        ax_trial.plot(
            trm_model, trial_centers, color="crimson", linestyle="--", linewidth=1.3
        )
        ax_trial.set_xlim(0, trial_xmax * 1.05)
        plt.setp(ax_trial.get_yticklabels(), visible=False)
        ax_trial.tick_params(labelsize=7)

        cax = fig.add_axes([0.89, 0.15, 0.02, 0.65])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label("Event Rate [Hz]", fontsize=10)

        return fig, np.array([ax_time, ax_emp, ax_mod, ax_trial])

    @staticmethod
    def plot_model_fits(
        dataset: PointProcessDataset,
        models: List[PointProcess],
        figsize: Tuple[int, int] = (12, 6),
    ) -> Tuple[plt.Figure, plt.Axes]:

        fig, ax = plt.subplots(figsize=figsize)

        # 1. Empirical PSTH
        ax.plot(
            dataset.t_centers,
            dataset.time_histogram_hz,
            color="black",
            linewidth=2.0,
            label="Empirical PSTH",
            zorder=4,  # Kept on top for visibility
        )

        # 2. Evaluate Models & Build Labels with LaTeX Formulas
        default_colors = plt.cm.tab10.colors

        for idx, model in enumerate(models):
            pred_surface = model.compute_expected_rate(dataset)
            mean_pred = np.average(
                pred_surface, axis=0, weights=dataset.n_fish_per_trial
            )
            color = default_colors[idx % len(default_colors)]
            label = f"{model.name}:  {model.latex_formula}"
            ax.plot(
                dataset.t_centers,
                mean_pred,
                linestyle="--",
                linewidth=1.8,
                color=color,
                label=label,
                zorder=3,
            )

        # 3. Plot Formatting
        ax.set_title(
            f"{dataset.category_label} (Laterality: {dataset.laterality})",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Time in Trial (s)", fontsize=11)
        ax.set_ylabel("Rate [Hz]", fontsize=11)
        ax.set_xlim(dataset.t_grid[0], dataset.t_grid[-1])
        ax.set_ylim(bottom=0.0)
        ax.grid(True, linestyle=":", alpha=0.6)

        # 4. Legend below the axes
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=1,  # 1 column so long formulas don't overlap horizontally
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            fontsize=10,
        )

        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_trial_traces(
        dataset: PointProcessDataset,
        model: PointProcess,
        trial_step: int = 2,
        figsize: Tuple[int, int] = (12, 6),
        cmap: str = "viridis",
        marker_size: float = 28.0,
        data_alpha: float = 0.35,
        model_alpha: float = 1.0,
    ) -> Tuple[plt.Figure, plt.Axes]:

        fig, ax = plt.subplots(figsize=figsize)

        model_surface = model.compute_expected_rate(dataset)

        # 2. Color Mapping Setup
        norm = plt.Normalize(vmin=0, vmax=dataset.num_trials)
        base_cmap = plt.get_cmap(cmap)

        # 3. Plot Selected Trials
        selected_indices = range(0, dataset.num_trials, trial_step)

        for idx in selected_indices:
            color = base_cmap(norm(idx))

            # Empirical Data (Markers Only, Semi-Transparent, Plotted Underneath)
            ax.scatter(
                dataset.t_centers,
                dataset.time_trial_histogram_hz[idx, :],
                color=color,
                s=marker_size,
                alpha=data_alpha,
                linewidths=0,
                zorder=2,
            )

            # Model Fits (Continuous Dashed Line, Plotted On Top)
            ax.plot(
                dataset.t_centers,
                model_surface[idx, :],
                color=color,
                linestyle="--",
                linewidth=1.8,
                alpha=model_alpha,
                zorder=3,
            )

        # 4. Legend to clarify Data vs. Model styling
        ax.scatter(
            [],
            [],
            color="gray",
            s=marker_size,
            alpha=0.5,
            label="Empirical Data (Points)",
        )
        ax.plot(
            [],
            [],
            color="gray",
            linestyle="--",
            linewidth=1.8,
            label="Model Fit (Dashed Line)",
        )

        # Plot Formatting
        ax.set_title(
            f"Trial-by-Trial Overlay | {model.name} (Every {trial_step} Trials)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Time in Trial (s)", fontsize=11)
        ax.set_ylabel("Rate [Hz]", fontsize=11)
        ax.set_xlim(dataset.t_grid[0], dataset.t_grid[-1])
        ax.set_ylim(bottom=0.0)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

        # 5. Colorbar for Trial Progression
        sm = plt.cm.ScalarMappable(cmap=base_cmap, norm=norm)
        sm.set_array([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.12)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Trial Index", fontsize=10)

        plt.tight_layout()
        return fig, ax
