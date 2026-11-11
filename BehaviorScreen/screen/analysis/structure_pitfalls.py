"""
Why treatment vs control comparisons must respect the structure of the data,
for a non-specialist audience: one figure per mechanism, read left to right
(the situation, then what a simple analysis makes of it). Numbers like
the prey-capture control fits: about 5.5 J-turns in a 24 s trial.

    python structure_pitfalls.py  ->  structure_1_history.png, structure_2_trials.png,
                                      structure_3_time.png, structure_4_timing.png
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import trapezoid

rng = np.random.default_rng(2)
DURATION, N_TRIALS = 24.0, 10
MEAN_BOUTS = 5.5  # J-turns per trial
CONTROL, TREATMENT, OTHER, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#262625", "#5f5e58"


def style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")


def headline(fig, title: str, subtitle: str) -> None:
    fig.text(0.01, 0.98, title, fontsize=15, fontweight="bold", va="top", color=INK)
    fig.text(0.01, 0.915, subtitle, fontsize=11, va="top", color=MUTED)


def bars(ax, labels: list, values: list, colors: list, ylabel: str, fmt: str = "{:.0f}%") -> None:
    ax.bar(labels, values, color=colors, width=0.6)
    top = max(max(values), 1)
    for i, v in enumerate(values):
        ax.text(i, max(v, 0) + 0.02 * top, fmt.format(v), ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
    ax.set_ylabel(ylabel)


def save(fig, name: str) -> None:
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(name, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 1. bout history
HUNT_RATE = lambda t: 0.4 * np.exp(-t / 1.2) + 0.08  # hunts started per second (control): the response to the prey
PER_HUNT, J_GAP = 2.5, 0.4  # J-turns per hunt (control mean); seconds between the J-turns of a hunt
LINES = {"control": (1.0, PER_HUNT, CONTROL), "line X: starts more hunts": (1.5, PER_HUNT, TREATMENT),
         "line Y: keeps each hunt going longer": (1.0, 1.5 * PER_HUNT, OTHER)}


def hunting_trial(scale: float, per_hunt: float) -> np.ndarray:
    """J-turn times in one trial: hunts start at the (scaled) prey-driven rate; each holds a random number of J-turns."""
    t = np.arange(0, DURATION, 0.01)
    starts = t[rng.random(len(t)) < scale * HUNT_RATE(t) * 0.01]
    sizes = rng.geometric(1 / per_hunt, len(starts))
    times = np.concatenate([s + J_GAP * np.arange(n) + rng.normal(0, 0.05, n) for s, n in zip(starts, sizes)] or [np.empty(0)])
    return np.sort(times[(times >= 0) & (times < DURATION)])


def typical_trial(scale: float, per_hunt: float) -> np.ndarray:
    """An illustrative trial in the first 12 s: a typical number of hunts, each of a typical length."""
    t = np.linspace(0, 12, 1201)
    weights = HUNT_RATE(t) / HUNT_RATE(t).sum()
    starts = np.sort(rng.choice(t, round(2 * scale), replace=False, p=weights))
    while np.any(np.diff(starts) < 2.5):  # hunts that do not run into each other
        starts = np.sort(rng.choice(t, round(2 * scale), replace=False, p=weights))
    sizes = rng.integers(round(per_hunt) - 1, round(per_hunt) + 2, len(starts))
    return np.concatenate([s + J_GAP * np.arange(n) for s, n in zip(starts, sizes)])


def figure_history() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.15, 1]})
    headline(fig, "1. Why the model needs bout history: two different effects, one J-turn rate",
             "Line X starts more hunts; line Y keeps each hunt going longer. Both make 50% more J-turns at the same moments, so a model of the J-turn\n"
             "rate alone says the same thing about both: 'responds 50% more to prey'. Explaining each J-turn by the ones before it tells them apart.")

    ax = axes[0]
    for group, (name, (scale, per_hunt, color)) in enumerate(LINES.items()):
        base = (len(LINES) - 1 - group) * 5.5
        for row in range(4):
            ax.vlines(typical_trial(scale, per_hunt), base + row + 0.15, base + row + 0.85, color=color, lw=1.5)
    ax.set_yticks([(len(LINES) - 1 - g) * 5.5 + 2 for g in range(len(LINES))], ["control", "line X", "line Y"])
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 12); ax.set_xlabel("time since the prey appeared (s); one row per trial, one tick per J-turn")
    style(ax, "a. What the fish do")

    ax = axes[1]
    edges = np.arange(0, DURATION + 0.5, 0.5)
    for name, (scale, per_hunt, color) in LINES.items():
        counts = np.histogram(np.concatenate([hunting_trial(scale, per_hunt) for _ in range(4000)]), edges)[0]
        ax.step(edges[:-1], counts / 4000 / 0.5, where="post", color=color, lw=2.2, label=name.split(":")[0])
    ax.set_xlim(0, 12); ax.set_ylim(0, None)
    ax.set_xlabel("time since the prey appeared (s)"); ax.set_ylabel("J-turns per second")
    ax.text(11.7, 0.9 * ax.get_ylim()[1], "X and Y: almost the same curve,\nboth 50% above control", ha="right", va="top", fontsize=10, color=INK)
    ax.legend(frameon=False, loc="center right")
    style(ax, "b. Without bout history: X and Y look the same")

    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig("structure_1_history.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 2. trials: lower response or faster habituation
def figure_trials() -> None:
    m = np.arange(N_TRIALS)
    control = MEAN_BOUTS * np.exp(-0.1 * m)
    lower = 0.75 * control  # responds less, gets used to the stimulus as usual
    rate = next(r for r in np.linspace(0.1, 1, 9001) if np.mean(MEAN_BOUTS * np.exp(-r * m)) <= lower.mean())
    faster = MEAN_BOUTS * np.exp(-rate * m)  # responds as usual at first, gets used to it faster

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.3, 1]})
    headline(fig, "2. 'Responds less' and 'gets used to it faster' can give the same average",
             "Fish respond less as the stimulus repeats. Averaging over all trials mixes how strongly a fish responds "
             "with how fast it stops responding.")
    groups = (("control", control, CONTROL, 0.1), ("treatment A", lower, TREATMENT, 0.1), ("treatment B", faster, OTHER, rate))

    ax = axes[0]
    for (name, y, color, _), label in zip(groups, ("control", "treatment A: responds less", "treatment B: gets used to it faster")):
        ax.plot(m, y, color=color, lw=2.5, marker="o", ms=5, label=label)
    ax.set_xlabel("trial (stimulus repetition)"); ax.set_ylabel("average J-turns in a trial"); ax.set_ylim(0, 6.5)
    ax.legend(frameon=False, fontsize=9)
    style(ax, "a. Two different effects")

    names, colors = [g[0] for g in groups], [g[2] for g in groups]
    ax = axes[1]
    bars(ax, names, [y.mean() for _, y, _, _ in groups], colors, "J-turns in a trial, averaged over the 10 trials", fmt="{:.1f}")
    ax.set_ylim(0, 6.5)
    style(ax, "b. The average over trials cannot tell them apart")

    save(fig, "structure_2_trials.png")


# ---------------------------------------------------------------- 3. time: a change diluted in the whole-trial count
A, TAU, B = 0.8, 1.2, 0.2


def figure_dilution() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    headline(fig, "3. Counting over the whole trial drowns a change in the response",
             "The treatment halves the burst of J-turns right after the stimulus starts; the steady activity is unchanged. "
             "Over 24 s the steady activity dominates the count.")

    ax = axes[0]
    ts = np.linspace(0, DURATION, 500)
    ax.plot(ts, A * np.exp(-ts / TAU) + B, color=CONTROL, lw=2.5, label="control")
    ax.plot(ts, A / 2 * np.exp(-ts / TAU) + B, color=TREATMENT, lw=2.5, label="treatment")
    ax.set_xlabel("time since the stimulus started (s)"); ax.set_ylabel("J-turns per second"); ax.set_xlim(0, DURATION); ax.set_ylim(0, 1.1)
    ax.annotate("response to the\nstimulus onset", (0.8, 0.75), (4, 0.85), fontsize=10, color=INK, arrowprops={"arrowstyle": "->", "color": MUTED})
    ax.text(14, 0.27, "steady activity", fontsize=10, color=INK)
    ax.legend(frameon=False, loc="upper right")
    style(ax, "a. Only the onset response changes")

    ax = axes[1]
    onset = lambda a: a * TAU * (1 - np.exp(-DURATION / TAU))
    for y, (a, color) in enumerate(((A / 2, TREATMENT), (A, CONTROL))):
        ax.barh(y, B * DURATION, color=color, alpha=0.35, height=0.55)
        ax.barh(y, onset(a), left=B * DURATION, color=color, height=0.55)
        ax.text(B * DURATION + onset(a) + 0.1, y, f"{B * DURATION + onset(a):.1f}", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_yticks([0, 1], ["treatment", "control"]); ax.set_xlim(0, 7.5)
    ax.text(B * DURATION / 2, 1.45, "steady activity", ha="center", fontsize=10, color=MUTED)
    ax.text(B * DURATION + 0.5, 1.45, "onset", ha="center", fontsize=10, color=MUTED)
    ax.set_xlabel("average J-turns in a trial"); ax.set_ylim(-0.5, 1.7)
    style(ax, "b. A halved response is an 8% smaller count")

    save(fig, "structure_3_time.png")


# ---------------------------------------------------------------- 4. time: the same count from different timing
def figure_timing() -> None:
    ts = np.linspace(0, DURATION, 2401)
    control = A * np.exp(-ts / TAU) + B  # quick response at the onset
    delay, width = 5.0, 2.5
    late = stats.gamma.pdf(ts, (delay / width) ** 2, scale=width ** 2 / delay)
    treatment = A * TAU * (1 - np.exp(-DURATION / TAU)) * late / trapezoid(late, ts) + B  # as many J-turns, later and slower
    cumulative = lambda y: np.concatenate([[0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(ts))])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    headline(fig, "4. Responding early or late can give exactly the same count",
             "The treated fish respond as much, but seconds later and more slowly. A count over the trial sees no difference; "
             "the timing of the response is what changed.")

    ax = axes[0]
    ax.plot(ts, control, color=CONTROL, lw=2.5, label="control: quick response")
    ax.plot(ts, treatment, color=TREATMENT, lw=2.5, label="treatment: slow, late response")
    ax.set_xlabel("time since the stimulus started (s)"); ax.set_ylabel("J-turns per second")
    ax.set_xlim(0, 15); ax.set_ylim(0, 1.1)
    ax.legend(frameon=False)
    style(ax, "a. Different timing")

    ax = axes[1]
    for y, color in ((control, CONTROL), (treatment, TREATMENT)):
        ax.plot(ts, cumulative(y), color=color, lw=2.5)
    ax.text(DURATION - 0.3, cumulative(control)[-1] + 0.3, f"both: {cumulative(control)[-1]:.1f} J-turns", ha="right", fontsize=11,
            fontweight="bold", color=INK)
    ax.set_xlabel("time since the stimulus started (s)"); ax.set_ylabel("J-turns so far in the trial")
    ax.set_xlim(0, DURATION); ax.set_ylim(0, 7.5)
    style(ax, "b. The running counts meet at the end")

    save(fig, "structure_4_timing.png")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10})
    figure_history()
    figure_trials()
    figure_dilution()
    figure_timing()
