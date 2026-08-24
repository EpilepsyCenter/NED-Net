#!/usr/bin/env python
"""Build Supplementary Figure 3 — NED-Net detection performance.

Panels
------
a  Representative automatic detection of a convulsive seizure: EEG trace with
   the U-Net per-sample seizure probability beneath it, the 0.5 detection and
   0.1 boundary thresholds marked, and the accepted event shaded.
b  The same for a non-convulsive seizure.
c  Representative rule-based interictal-spike detections.
d  Blinded spot-check on study recordings (precision).
e  Detector confidence for spot-check detections the expert confirmed versus
   rejected.
f  Convulsive classifier confusion matrix on held-out validation animals.
g  Event-level performance versus detection threshold (written by
   ``threshold_sweep.json`` if present; skipped otherwise).

    python scripts/paper_stats/supp_fig3.py

Writes supp_fig3.pdf / .svg / .png next to this script.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).parent

# Validated categorical palette (dataviz skill reference instance, slots 1-3).
# node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
TRACE = "#2f2f2e"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.edgecolor": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "pdf.fonttype": 42,   # editable text in Illustrator
    "svg.fonttype": "none",
})


def _clean(ax, left=True, bottom=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.set_yticks([])
    if not bottom:
        ax.set_xticks([])


def _panel_label(ax, letter, dx=-0.085, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left", color=INK)


# ---------------------------------------------------------------------------
# Panels a / b — trace + probability
# ---------------------------------------------------------------------------

def trace_panel(ax_eeg, ax_prob, tr, key, title, conv_prob, letter):
    fs = int(tr["fs"][0])
    sig = tr[f"{key}_signal"]
    prob = tr[f"{key}_prob"]
    ev0, ev1 = tr[f"{key}_event"]
    t = np.arange(len(sig)) / fs

    for ax in (ax_eeg, ax_prob):
        ax.axvspan(ev0, ev1, color=BLUE, alpha=0.10, lw=0, zorder=0)

    ax_eeg.plot(t, sig, color=TRACE, lw=0.35, zorder=2)
    _clean(ax_eeg, left=False, bottom=False)
    ax_eeg.spines["bottom"].set_visible(False)
    ax_eeg.set_xlim(t[0], t[-1])
    ax_eeg.set_title(title, loc="left", color=INK, pad=3)
    _panel_label(ax_eeg, letter, dy=1.14)

    # amplitude scale bar, clear of the trace on the right
    amp = float(np.percentile(np.abs(sig), 99.5))
    lo, hi = ax_eeg.get_ylim()
    ax_eeg.set_ylim(lo, hi)
    xb = t[-1] * 1.012
    ax_eeg.plot([xb, xb], [0, amp], color=INK2, lw=1.2,
                solid_capstyle="butt", clip_on=False)
    ax_eeg.text(xb + t[-1] * 0.006, amp / 2, f"{amp:.1f} mV", ha="left",
                va="center", fontsize=5.5, color=INK2, clip_on=False)

    ax_prob.plot(t[:len(prob)], prob, color=BLUE, lw=0.9, zorder=3)
    ax_prob.axhline(0.5, color=INK2, lw=0.5, zorder=1)
    ax_prob.axhline(0.1, color=MUTED, lw=0.5, zorder=1)
    ax_prob.text(t[-1], 0.5, " 0.5 detection", va="center", ha="left",
                 fontsize=5.5, color=INK2, clip_on=False)
    ax_prob.text(t[-1], 0.1, " 0.1 boundary", va="center", ha="left",
                 fontsize=5.5, color=MUTED, clip_on=False)
    ax_prob.set_ylim(-0.04, 1.04)
    ax_prob.set_yticks([0, 0.5, 1])
    ax_prob.set_xlim(t[0], t[-1])
    ax_prob.set_ylabel("P(seizure)", fontsize=6.5)
    ax_prob.set_xlabel("Time (s)")
    _clean(ax_prob)

    ax_prob.text(0.995, 0.97,
                 f"detected {ev1 - ev0:.1f} s\nStage-2 P(convulsive) = {conv_prob:.3f}",
                 transform=ax_prob.transAxes, ha="right", va="top",
                 fontsize=5.8, color=INK2, linespacing=1.4)


# ---------------------------------------------------------------------------
# Panel c — interictal spikes
# ---------------------------------------------------------------------------

def spike_panel(ax, tr, letter):
    fs = int(tr["fs"][0])
    sig = tr["spike_signal"]
    times = tr["spike_times"]
    t = np.arange(len(sig)) / fs
    ax.plot(t, sig, color=TRACE, lw=0.35, zorder=2)
    lo, hi = float(sig.min()), float(sig.max())
    pad = (hi - lo) * 0.30
    ax.set_ylim(lo - pad * 0.4, hi + pad)
    for x in times:
        ax.plot([x], [hi + pad * 0.45], marker="v", ms=3.2, color=ORANGE,
                mec="none", zorder=3)
    _clean(ax, left=False, bottom=False)
    ax.set_xlim(t[0], t[-1])
    ax.set_title("Interictal spikes — rule-based detector", loc="left",
                 color=INK, pad=10)
    _panel_label(ax, letter, dy=1.16)
    ax.text(1.0, 1.005, f"{len(times)} detections", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color=ORANGE)
    y0 = ax.get_ylim()[0]
    ax.plot([t[-1] - 2, t[-1]], [y0, y0], color=INK2, lw=1.2,
            solid_capstyle="butt", clip_on=False)
    ax.text(t[-1] - 1, y0, "2 s", ha="center", va="bottom", fontsize=6, color=INK2)


# ---------------------------------------------------------------------------
# Panel d — spot-check precision (stat tile, not a 2-slice pie)
# ---------------------------------------------------------------------------

def precision_panel(ax, n_conf, n_rej, letter):
    total = n_conf + n_rej
    prec = 100 * n_conf / total
    ax.axis("off")
    _panel_label(ax, letter, dx=-0.02, dy=1.06)
    ax.text(0, 0.74, f"{prec:.1f}%", fontsize=21, color=BLUE,
            fontweight="bold", va="top", ha="left", transform=ax.transAxes)
    ax.text(0, 0.44, "precision, blinded spot-check", fontsize=7, color=INK,
            va="top", ha="left", transform=ax.transAxes)
    ax.text(0, 0.31, f"{n_conf} of {total} detections confirmed\n"
                     "30 recordings, expert re-read",
            fontsize=6.5, color=INK2, va="top", ha="left", transform=ax.transAxes)
    # composition bar with a 2px surface gap between segments
    bar = ax.inset_axes([0, 0.02, 1.0, 0.10])
    bar.barh([0], [n_conf], color=BLUE, height=1.0)
    bar.barh([0], [n_rej], left=n_conf + total * 0.004, color=ORANGE, height=1.0)
    bar.set_xlim(0, total)
    bar.axis("off")
    bar.text(n_conf / 2, 0, f"{n_conf} confirmed", ha="center", va="center",
             fontsize=5.5, color="white")
    bar.text(n_conf + n_rej / 2, -1.5, f"{n_rej} rejected", ha="center",
             va="top", fontsize=5.5, color=ORANGE)


# ---------------------------------------------------------------------------
# Panel e — confidence separation
# ---------------------------------------------------------------------------

def confidence_panel(ax, conf, rej, letter):
    rng = np.random.default_rng(0)
    for i, (vals, color, label) in enumerate(
            ((conf, BLUE, "confirmed"), (rej, ORANGE, "rejected"))):
        x = i + rng.uniform(-0.13, 0.13, len(vals))
        ax.scatter(x, vals, s=7, color=color, alpha=0.55, lw=0, zorder=2)
        med = np.median(vals)
        ax.plot([i - 0.28, i + 0.28], [med, med], color=color, lw=1.6,
                solid_capstyle="round", zorder=3)
        ax.text(i, 1.06, f"n = {len(vals)}", ha="center", va="bottom",
                fontsize=6, color=INK2)
        ax.text(i + 0.34, med, f"{med:.2f}", ha="left", va="center",
                fontsize=6, color=color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["confirmed", "rejected"])
    ax.set_ylabel("Detector confidence")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, 1.6)
    ax.set_title("Spot-check detections", loc="left", color=INK, pad=12)
    _clean(ax)
    _panel_label(ax, letter, dx=-0.20, dy=1.18)


# ---------------------------------------------------------------------------
# Panel f — convulsive classifier confusion matrix
# ---------------------------------------------------------------------------

def convulsive_panel(ax, cp, letter):
    """Stage-2 convulsive probability per validation crop, split by expert label.

    Replaces a confusion matrix: the same four counts are readable off the plot
    (points either side of the threshold line), but the separation between the
    classes and the location of the errors are visible too — which a matrix
    dominated by its true-negative cell hides.
    """
    prob = np.array(cp["prob"], float)
    label = np.array(cp["label"], int)
    thr = float(cp["threshold"])
    rng = np.random.default_rng(1)

    groups = ((0, "non-convulsive", ORANGE), (1, "convulsive", BLUE))
    for i, (lab, name, color) in enumerate(groups):
        v = prob[label == lab]
        x = i + rng.uniform(-0.15, 0.15, len(v))
        wrong = (v > thr) if lab == 0 else (v <= thr)
        ax.scatter(x[~wrong], v[~wrong], s=5, color=color, alpha=0.35, lw=0, zorder=2)
        # errors: same hue, ringed so they read as a distinct subset without a new colour
        ax.scatter(x[wrong], v[wrong], s=13, facecolor=color, edgecolor=INK,
                   linewidth=0.6, zorder=4)
        ax.text(i, 1.06, f"{name}\nn = {len(v)}", ha="center", va="bottom",
                fontsize=6, color=INK2, linespacing=1.3)

    ax.axhline(thr, color=INK2, lw=0.7, zorder=1)
    ax.text(1.62, thr, f" {thr:g}\n threshold", va="center", ha="left",
            fontsize=5.8, color=INK2, linespacing=1.3, clip_on=False)
    ax.annotate(f"{cp['fp']} false positives", xy=(0.19, 0.91),
                xytext=(0.33, 0.68), fontsize=5.8, color=INK2,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
    ax.annotate(f"{cp['fn']} missed", xy=(0.86, 0.11),
                xytext=(0.30, 0.27), fontsize=5.8, color=INK2,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))

    ax.set_ylabel("Stage-2 P(convulsive)")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.5, 1.6)
    ax.set_xticks([])
    ax.set_title("Convulsive classifier", loc="left", color=INK, pad=16)
    _clean(ax, bottom=False)
    _panel_label(ax, letter, dy=1.24)
    f1 = 2 * cp["tp"] / (2 * cp["tp"] + cp["fp"] + cp["fn"])
    ax.text(0.5, -0.10, f"F1 = {f1:.3f}   ·   held-out animals",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=6, color=INK2)


# ---------------------------------------------------------------------------
# Panel g — threshold sweep
# ---------------------------------------------------------------------------

def sweep_panel(ax, sweep, letter):
    thr = np.array(sweep["thresholds"])
    series = [("Recall", sweep["recall"], ORANGE),
              ("Precision", sweep["precision"], BLUE),
              ("F1", sweep["f1"], AQUA)]
    for name, vals, color in series:
        vals = np.array(vals)
        ax.plot(thr, vals, color=color, lw=1.4, zorder=2)
        # direct label (relief rule: aqua is below 3:1 on the light surface)
        ax.text(thr[-1] + 0.012, vals[-1], name, color=color, fontsize=6.5,
                va="center", ha="left")
    ax.axvline(0.5, color=INK2, lw=0.5, zorder=1)
    ax.text(0.5, 1.02, "operating point", fontsize=5.5, color=INK2,
            ha="center", va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Detection threshold")
    ax.set_ylabel("Event-level score")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_title("Held-out validation animals", loc="left", color=INK, pad=10)
    _clean(ax)
    _panel_label(ax, letter)


# ---------------------------------------------------------------------------

def main() -> None:
    tr = np.load(HERE / "traces.npz")

    spot = json.loads((HERE / "spotcheck.json").read_text())
    sweep_path = HERE / "threshold_sweep.json"
    sweep = json.loads(sweep_path.read_text()) if sweep_path.exists() else None

    fig = plt.figure(figsize=(7.2, 8.2))
    fig.patch.set_facecolor("white")
    outer = GridSpec(
        4, 1, figure=fig, height_ratios=[1.42, 1.42, 1.10, 1.15],
        hspace=0.50, left=0.085, right=0.935, top=0.945, bottom=0.055)

    for row, (key, title, cp, letter) in enumerate((
            ("convulsive", "Convulsive seizure — automatic detection", 0.998, "a"),
            ("nonconvulsive", "Non-convulsive seizure — automatic detection", 0.291, "b"))):
        sub = outer[row].subgridspec(2, 1, height_ratios=[1.0, 0.72], hspace=0.10)
        trace_panel(fig.add_subplot(sub[0]), fig.add_subplot(sub[1]),
                    tr, key, title, cp, letter)

    mid = outer[2].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30)
    spike_panel(fig.add_subplot(mid[0]), tr, "c")
    precision_panel(fig.add_subplot(mid[1]), spot["n_confirmed"],
                    spot["n_rejected"], "d")

    bot = outer[3].subgridspec(1, 3, width_ratios=[1.0, 1.30, 1.05], wspace=0.52)
    confidence_panel(fig.add_subplot(bot[0]),
                     np.array(spot["conf_confirmed"]),
                     np.array(spot["conf_rejected"]), "e")
    cp = json.loads((HERE / "convulsive_probs.json").read_text())
    if sweep:
        sweep_panel(fig.add_subplot(bot[1]), sweep, "f")
        convulsive_panel(fig.add_subplot(bot[2]), cp, "g")
    else:
        convulsive_panel(fig.add_subplot(bot[1]), cp, "f")

    for ext in ("pdf", "svg", "png"):
        fig.savefig(HERE / f"supp_fig3.{ext}", dpi=400,
                    facecolor="white", bbox_inches="tight")
    print("wrote", ", ".join(str(HERE / f"supp_fig3.{e}") for e in ("pdf", "svg", "png")))


if __name__ == "__main__":
    main()
