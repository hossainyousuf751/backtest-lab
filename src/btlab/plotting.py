"""Diagnostic plots (requires matplotlib)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btlab import metrics

C = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def plot_equity(series: dict[str, pd.Series], path: str, title: str):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    styles = ["-", "--", ":", "-."]
    for i, (name, r) in enumerate(series.items()):
        eq = (1 + r).cumprod()
        ax1.plot(eq.index, eq, styles[i % 4], color=C[i % 5], lw=1.6,
                 label=f"{name} (Sharpe {metrics.sharpe(r):.2f})")
        ax2.plot(eq.index, eq / eq.cummax() - 1, styles[i % 4], color=C[i % 5], lw=1.2)
    ax1.set_ylabel("growth of 1")
    ax1.legend(frameon=False)
    ax2.set_ylabel("drawdown")
    ax1.set_title(title, fontweight="bold", loc="left")
    _style(ax1), _style(ax2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trial_sharpes(trial_sharpes, expected_max: float, path: str, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(trial_sharpes, bins=25, color=C[0], edgecolor="white", alpha=0.85)
    best = float(np.nanmax(trial_sharpes))
    ax.axvline(best, color=C[3], ls="-", lw=2, label=f"best trial: {best:.2f}")
    ax.axvline(expected_max, color=C[1], ls="--", lw=2,
               label=f"expected best of {len(trial_sharpes)} with zero skill: {expected_max:.2f}")
    ax.set_xlabel("annualised Sharpe ratio (full sample)")
    ax.set_ylabel("number of configurations")
    ax.legend(frameon=False)
    ax.set_title(title, fontweight="bold", loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_walk_forward(folds: pd.DataFrame, path: str, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(folds))
    ax.bar(x - 0.2, folds["is_sharpe"], 0.4, color=C[0], label="in-sample (selected config)")
    ax.bar(x + 0.2, folds["oos_sharpe"], 0.4, color=C[1], hatch="//", edgecolor="white",
           label="out-of-sample (next year)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [d.strftime("%Y") for d in folds["test_start"]])
    ax.set_xlabel("test year")
    ax.set_ylabel("annualised Sharpe")
    ax.legend(frameon=False)
    ax.set_title(title, fontweight="bold", loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_capacity(curves: dict[str, pd.DataFrame], path: str, title: str):
    """Net Sharpe against fund size; each table has columns capital and sharpe."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, t) in enumerate(curves.items()):
        ax.plot(t["capital"], t["sharpe"], "os^D"[i % 4] + "-", color=C[i % 5], lw=2, label=name)
    ax.set_xscale("log")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("capital ($, log scale)")
    ax.set_ylabel("net annualised Sharpe")
    ax.legend(frameon=False)
    ax.set_title(title, fontweight="bold", loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost_sensitivity(table: pd.DataFrame, path: str, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(table["cost_bps"], table["sharpe"], "o-", color=C[0], lw=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("one-way trading cost (bps)")
    ax.set_ylabel("net annualised Sharpe")
    ax.set_title(title, fontweight="bold", loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
