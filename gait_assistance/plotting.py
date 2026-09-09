"""Result plots for the offline simulation (spec 34).

The PCA projection here is **visualisation only**.  Classification, clustering
and every control distance are computed in the full Log-Euclidean space; no
reduced representation ever reaches the controller.

Colour is assigned from a fixed, contrast-validated categorical order and is
always paired with a marker shape, so gait-state identity never depends on
colour alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .manifold.reference import MetricRange
from .offline_sim import SimulationResult

#: Categorical slots, fixed order, never cycled.
STATE_COLORS: Tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
#: Marker shapes carrying the same identity as the colour (secondary encoding).
STATE_MARKERS: Tuple[str, ...] = ("o", "s", "^", "D")
COLOR_OOD: str = "#d03b3b"        #: status/critical
COLOR_UNKNOWN: str = "#8a8a85"    #: muted ink
COLOR_BASELINE: str = "#b5b5ae"
TEXT_PRIMARY: str = "#0b0b0b"
TEXT_SECONDARY: str = "#52514e"
GRID_COLOR: str = "#e3e2de"
SERIES = {"patient": "#2a78d6", "healthy": "#eb6834", "target": "#1baf7a"}


def _style_axis(ax: "object", title: str, xlabel: str, ylabel: str) -> None:
    """Apply the recessive grid/axis treatment shared by every panel."""
    ax.set_title(title, fontsize=10, color=TEXT_PRIMARY, loc="left", pad=8)  # type: ignore[attr-defined]
    ax.set_xlabel(xlabel, fontsize=8.5, color=TEXT_SECONDARY)  # type: ignore[attr-defined]
    ax.set_ylabel(ylabel, fontsize=8.5, color=TEXT_SECONDARY)  # type: ignore[attr-defined]
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.9)  # type: ignore[attr-defined]
    ax.set_axisbelow(True)  # type: ignore[attr-defined]
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)  # type: ignore[attr-defined]
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)  # type: ignore[attr-defined]
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)  # type: ignore[attr-defined]


def state_style(label: str) -> Tuple[str, str]:
    """Return the ``(colour, marker)`` pair of a gait-state label."""
    if label == "OOD":
        return COLOR_OOD, "X"
    if label == "UNKNOWN":
        return COLOR_UNKNOWN, "P"
    try:
        index = int(label.lstrip("S"))
    except ValueError:
        return COLOR_UNKNOWN, "P"
    return STATE_COLORS[index % len(STATE_COLORS)], STATE_MARKERS[index % len(STATE_MARKERS)]


def pca_projection(
    vectors: np.ndarray, n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Project vectorised log-SPD matrices for display only.

    Args:
        vectors: ``(n_strides, d)`` matrix of vectorised ``log(C)``.
        n_components: number of principal components.

    Returns:
        ``(scores, explained_variance_ratio)``; empty arrays when there is
        not enough data to fit.
    """
    data = np.asarray(vectors, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2:
        return np.zeros((0, n_components)), np.zeros(n_components)
    k = min(n_components, data.shape[0], data.shape[1])
    pca = PCA(n_components=k)
    scores = pca.fit_transform(data)
    if scores.shape[1] < n_components:
        pad = np.zeros((scores.shape[0], n_components - scores.shape[1]))
        scores = np.hstack([scores, pad])
    ratio = np.zeros(n_components)
    ratio[: pca.explained_variance_ratio_.size] = pca.explained_variance_ratio_
    return scores, ratio


def plot_results(
    result: SimulationResult,
    out_path: Union[str, Path],
    *,
    title: str = "Gait assistance - offline replay",
    dpi: int = 140,
) -> Optional[Path]:
    """Render the nine diagnostic panels of a simulation run.

    Panels, in reading order: the log-SPD PCA with the healthy region, the
    healthy-region distance, the baseline distance, the swing ratio and the
    belt excursion against their reference intervals, the biomechanical
    deficit, the manifold context score, the assist gain, and the decision
    state timeline.

    The layout follows the causal chain the controller uses: what the gait
    looks like, how far it is from each reference, which of those the device
    can act on, and what it decided - so a reader can see a gain rise and walk
    backwards to the deficit that justified it.

    Args:
        result: finished simulation result.
        out_path: destination image file.
        title: figure title.
        dpi: output resolution.

    Returns:
        The written path, or ``None`` when the run produced no analysed stride.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = result.table
    if table.empty:
        return None

    figure, axes = plt.subplots(3, 3, figsize=(18.0, 12.5), facecolor="#fcfcfb")
    figure.suptitle(title, fontsize=13, color=TEXT_PRIMARY, x=0.02, ha="left")
    for ax in axes.ravel():
        ax.set_facecolor("#fcfcfb")

    _panel_pca(axes[0, 0], result)
    _panel_healthy_distance(axes[0, 1], table)
    _panel_patient_distance(axes[0, 2], table)
    _panel_metric(
        axes[1, 0], table, "swing_ratio", "Estimated swing ratio per stride",
        "swing ratio (estimated)", result.metric_ranges.get("swing_ratio"),
    )
    _panel_metric(
        axes[1, 1], table, "belt_excursion", "Belt excursion per stride",
        "excursion (mm)", result.metric_ranges.get("belt_excursion"),
    )
    _panel_series(
        axes[1, 2], table, "biomechanical_deviation",
        "Biomechanical deficit E_B - drives the assistance",
        "E_B", SERIES["patient"],
    )
    _panel_series(
        axes[2, 0], table, "manifold_deviation",
        "Manifold deviation E_R - severity context only",
        "E_R", SERIES["healthy"],
    )
    _panel_gain(axes[2, 1], table, result.config.assist.max_gain)
    _panel_decision_states(axes[2, 2], table)

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)
    return destination


def _panel_pca(ax: "object", result: SimulationResult) -> None:
    """Scatter of the log-SPD vectors in the first two principal components.

    Baseline strides are drawn as open markers coloured by their cluster, the
    replayed strides as filled markers coloured by the state they were assigned
    online, and the cluster centroids as stars.  All of them are projected with
    one PCA fitted on the union, so the positions are comparable.
    """
    online = result.log_vectors
    valid_table = result.table[result.table["valid"] == 1]
    baseline = (
        result.model.baseline.vectors
        if result.model is not None
        else np.zeros((0, online.shape[1] if online.size else 0))
    )
    stacked = [v for v in (baseline, online) if v.size]
    if not stacked:
        _style_axis(ax, "PCA of log-SPD vectors (visualisation only)", "PC1", "PC2")
        return
    union = np.vstack(stacked)
    if union.shape[0] < 2:
        _style_axis(ax, "PCA of log-SPD vectors (visualisation only)", "PC1", "PC2")
        return
    pca = PCA(n_components=min(2, union.shape[0], union.shape[1])).fit(union)
    ratio = np.zeros(2)
    ratio[: pca.explained_variance_ratio_.size] = pca.explained_variance_ratio_

    seen: Dict[str, bool] = {}
    if baseline.size and result.model is not None:
        scores = _project(pca, baseline)
        for point, label_id in zip(scores, result.model.clusters.labels):
            label = f"S{int(label_id)}"
            color, marker = state_style(label)
            key = f"baseline {label}"
            ax.scatter(  # type: ignore[attr-defined]
                point[0], point[1], s=34, facecolors="none", edgecolors=color,
                linewidths=1.3, marker=marker, zorder=3,
                label=None if key in seen else key,
            )
            seen[key] = True
    if online.size:
        scores = _project(pca, online)
        labels = [str(s) for s in valid_table["gait_state"]][: scores.shape[0]]
        for point, label in zip(scores, labels):
            color, marker = state_style(label)
            key = f"online {label}"
            ax.scatter(  # type: ignore[attr-defined]
                point[0], point[1], s=48, c=color, edgecolors="#fcfcfb",
                linewidths=0.9, marker=marker, zorder=4,
                label=None if key in seen else key,
            )
            seen[key] = True
    if result.model is not None:
        centroids = _project(pca, result.model.clusters.centroid_vectors)
        ax.scatter(  # type: ignore[attr-defined]
            centroids[:, 0], centroids[:, 1], s=150, marker="*", c=TEXT_PRIMARY,
            label="patient baseline centroid", zorder=5,
        )
    _draw_healthy_region(ax, pca, result)
    _style_axis(
        ax,
        f"PCA of log-SPD vectors - display only ({ratio[0]:.0%}/{ratio[1]:.0%} var.)",
        "PC1",
        "PC2",
    )
    ax.margins(0.12)  # type: ignore[attr-defined]
    # An opaque frame: the healthy-region outline runs behind the legend and
    # would otherwise strike through the labels.
    ax.legend(  # type: ignore[attr-defined]
        frameon=True, facecolor="#fcfcfb", edgecolor=GRID_COLOR, framealpha=0.95,
        fontsize=7.5, labelcolor=TEXT_SECONDARY, loc="best", ncol=2,
    )


def _draw_healthy_region(ax: "object", pca: PCA, result: SimulationResult) -> None:
    """Overlay the healthy centroid and the outline of the healthy region.

    The region is a ball in the full Log-Euclidean space.  PCA projects with
    orthonormal components, so that ball projects *inside* a disc of the same
    radius: the circle drawn here is the outline of that projection, an upper
    bound rather than the exact boundary.  The control decision uses the full
    space, never this picture, and the label says so.
    """
    healthy = result.model.healthy if result.model is not None else None
    if healthy is None:
        return
    centre = _project(pca, healthy.centroid_vector.reshape(1, -1))[0]
    ax.scatter(  # type: ignore[attr-defined]
        centre[0], centre[1], s=170, marker="*", c=SERIES["healthy"],
        edgecolors="#fcfcfb", linewidths=0.9, label="healthy centroid", zorder=6,
    )
    radius = float(healthy.manifold_distance_threshold)
    if radius > 0.0:
        import matplotlib.patches as patches

        ax.add_patch(  # type: ignore[attr-defined]
            patches.Circle(
                (centre[0], centre[1]), radius, fill=False,
                edgecolor=SERIES["healthy"], linestyle="--", linewidth=1.4,
                label="healthy region (projected)", zorder=2,
            )
        )


def _panel_healthy_distance(ax: "object", table: pd.DataFrame) -> None:
    """Distance to the healthy centroid with the region boundary drawn."""
    if not table["d_healthy"].notna().any():
        _style_axis(
            ax, "Healthy-region distance - no healthy reference loaded",
            "stride id", "d_healthy",
        )
        ax.text(  # type: ignore[attr-defined]
            0.5, 0.5, "no healthy reference", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color=TEXT_SECONDARY,
        )
        return
    ax.plot(table["stride_id"], table["d_healthy"], color=SERIES["healthy"],  # type: ignore[attr-defined]
            linewidth=2.0, marker="s", markersize=4, label="d_healthy", zorder=3)
    threshold = table["healthy_region_threshold"].dropna()
    if not threshold.empty and np.isfinite(threshold.iloc[0]):
        level = float(threshold.iloc[0])
        ax.axhline(level, color=COLOR_UNKNOWN, linewidth=1.3, linestyle="--", zorder=2)  # type: ignore[attr-defined]
        ax.annotate(  # type: ignore[attr-defined]
            "healthy region boundary", xy=(0.99, level),
            xycoords=("axes fraction", "data"), ha="right", va="bottom",
            fontsize=8, color=TEXT_SECONDARY,
        )
    _style_axis(
        ax, "Distance to the healthy region", "stride id", "d_healthy (Frobenius)"
    )


def _panel_patient_distance(ax: "object", table: pd.DataFrame) -> None:
    """Distance to the session baseline: a trend, not an assist trigger."""
    ax.plot(table["stride_id"], table["d_patient"], color=SERIES["patient"],  # type: ignore[attr-defined]
            linewidth=2.0, marker="o", markersize=4, zorder=3)
    baseline = table[table["is_baseline"] == 1]["d_patient"]
    reference = baseline.mean() if not baseline.empty else table["d_patient"].mean()
    if np.isfinite(reference):
        ax.axhline(reference, color=COLOR_UNKNOWN, linewidth=1.2, linestyle="--", zorder=2)  # type: ignore[attr-defined]
        ax.annotate(  # type: ignore[attr-defined]
            "baseline mean", xy=(0.99, reference), xycoords=("axes fraction", "data"),
            ha="right", va="bottom", fontsize=8, color=TEXT_SECONDARY,
        )
    _style_axis(
        ax, "Distance to the session baseline - trend only, not a trigger",
        "stride id", "d_patient (Frobenius)",
    )


def _panel_series(
    ax: "object", table: pd.DataFrame, column: str, title: str, ylabel: str, color: str
) -> None:
    """One normalised score over the strides, on a fixed 0-1 axis."""
    if column not in table:
        _style_axis(ax, title, "stride id", ylabel)
        return
    ax.plot(table["stride_id"], table[column], color=color,  # type: ignore[attr-defined]
            linewidth=2.0, marker="o", markersize=4, zorder=3)
    ax.set_ylim(-0.05, 1.05)  # type: ignore[attr-defined]
    _style_axis(ax, title, "stride id", ylabel)


def _panel_decision_states(ax: "object", table: pd.DataFrame) -> None:
    """Timeline of the per-stride decision state."""
    order = [
        "IN_RANGE",
        "BIOMECH_DEFICIT_ONLY",
        "MANIFOLD_DEVIATION_ONLY",
        "COMBINED_DEVIATION",
        "OOD",
    ]
    present = [s for s in order if (table["decision_state"] == s).any()]
    positions = {label: i for i, label in enumerate(present)}
    palette = {
        "IN_RANGE": SERIES["target"],
        "BIOMECH_DEFICIT_ONLY": STATE_COLORS[0],
        "MANIFOLD_DEVIATION_ONLY": COLOR_UNKNOWN,
        "COMBINED_DEVIATION": STATE_COLORS[1],
        "OOD": COLOR_OOD,
    }
    markers = {
        "IN_RANGE": "o",
        "BIOMECH_DEFICIT_ONLY": "s",
        "MANIFOLD_DEVIATION_ONLY": "^",
        "COMBINED_DEVIATION": "D",
        "OOD": "X",
    }
    for label in present:
        subset = table[table["decision_state"] == label]
        ax.scatter(  # type: ignore[attr-defined]
            subset["stride_id"], [positions[label]] * len(subset),
            c=palette[label], marker=markers[label], s=46,
            edgecolors="#fcfcfb", linewidths=0.8, zorder=3,
        )
    ax.set_yticks(range(len(present)))  # type: ignore[attr-defined]
    ax.set_yticklabels(present, fontsize=7.5)  # type: ignore[attr-defined]
    ax.margins(y=0.35)  # type: ignore[attr-defined]
    # The tick labels name each state, so identity never rests on colour.
    _style_axis(ax, "Assistance decision state per stride", "stride id", "state")


def _project(pca: PCA, vectors: np.ndarray) -> np.ndarray:
    """Project ``vectors`` and pad to two columns when PCA kept only one."""
    scores = pca.transform(np.asarray(vectors, dtype=float))
    if scores.shape[1] < 2:
        scores = np.hstack([scores, np.zeros((scores.shape[0], 2 - scores.shape[1]))])
    return scores


def _panel_states(ax: "object", table: pd.DataFrame) -> None:
    """Gait state assigned to every analysed stride."""
    labels = sorted(set(str(s) for s in table["gait_state"]))
    positions = {label: i for i, label in enumerate(labels)}
    for label in labels:
        subset = table[table["gait_state"] == label]
        color, marker = state_style(label)
        ax.scatter(  # type: ignore[attr-defined]
            subset["stride_id"], [positions[label]] * len(subset),
            c=color, marker=marker, s=42, edgecolors="#fcfcfb", linewidths=0.8,
            label=label, zorder=3,
        )
    ax.set_yticks(range(len(labels)))  # type: ignore[attr-defined]
    ax.set_yticklabels(labels)  # type: ignore[attr-defined]
    ax.margins(y=0.35)  # type: ignore[attr-defined]
    # The y tick labels name every state directly, so identity never rests on
    # colour and a legend box would only overlap the marks.
    _style_axis(ax, "Gait state per stride", "stride id", "state")


def _panel_distances(ax: "object", table: pd.DataFrame) -> None:
    """Log-Euclidean distances to the three reference points."""
    ax.plot(table["stride_id"], table["d_patient"], color=SERIES["patient"],  # type: ignore[attr-defined]
            linewidth=2.0, marker="o", markersize=4, label="d_patient", zorder=3)
    if table["d_healthy"].notna().any():
        ax.plot(table["stride_id"], table["d_healthy"], color=SERIES["healthy"],  # type: ignore[attr-defined]
                linewidth=2.0, marker="s", markersize=4, label="d_healthy", zorder=3)
    ax.plot(table["stride_id"], table["d_target"], color=SERIES["target"],  # type: ignore[attr-defined]
            linewidth=2.0, marker="^", markersize=4, label="d_target", zorder=3)
    _style_axis(ax, "Log-Euclidean deviation per stride", "stride id", "distance (Frobenius)")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)  # type: ignore[attr-defined]


def _panel_gain(ax: "object", table: pd.DataFrame, max_gain: float) -> None:
    """Assist gain with the configured ceiling and the OOD strides marked."""
    ax.plot(table["stride_id"], table["assist_gain"], color=SERIES["patient"],  # type: ignore[attr-defined]
            linewidth=2.0, marker="o", markersize=4, zorder=3)
    ax.axhline(max_gain, color=COLOR_UNKNOWN, linewidth=1.2, linestyle="--", zorder=2)  # type: ignore[attr-defined]
    ax.annotate(  # type: ignore[attr-defined]
        f"max_gain {max_gain:g}", xy=(0.99, max_gain), xycoords=("axes fraction", "data"),
        ha="right", va="bottom", fontsize=8, color=TEXT_SECONDARY,
    )
    ood = table[table["ood_flag"] == 1]
    if not ood.empty:
        ax.scatter(ood["stride_id"], ood["assist_gain"], c=COLOR_OOD, marker="X",  # type: ignore[attr-defined]
                   s=60, label="OOD", zorder=4)
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)  # type: ignore[attr-defined]
    _style_axis(ax, "Assist gain per stride", "stride id", "assist gain")


def _panel_metric(
    ax: "object",
    table: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    metric_range: Optional["MetricRange"] = None,
) -> None:
    """One biomechanical metric with the reference interval it is judged against.

    The shaded band is the interval, not a target line: any value inside it
    scores zero error, which is exactly how the policy treats it.  Drawing a
    band instead of a mean is what keeps a reader from inferring that the
    median is something the device aims for.
    """
    ax.plot(table["stride_id"], table[column], color=SERIES["patient"],  # type: ignore[attr-defined]
            linewidth=2.0, marker="o", markersize=4, zorder=3)
    if metric_range is not None:
        ax.axhspan(  # type: ignore[attr-defined]
            metric_range.lower, metric_range.upper, color=SERIES["target"],
            alpha=0.13, zorder=1,
        )
        for level, name in (
            (metric_range.lower, "healthy lower"),
            (metric_range.upper, "healthy upper"),
        ):
            ax.axhline(level, color=SERIES["target"], linewidth=1.2,  # type: ignore[attr-defined]
                       linestyle="--", zorder=2)
            ax.annotate(  # type: ignore[attr-defined]
                name, xy=(0.99, level), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=7.5, color=TEXT_SECONDARY,
            )
    else:
        baseline = table[table["is_baseline"] == 1][column]
        reference = baseline.mean() if not baseline.empty else table[column].mean()
        if np.isfinite(reference):
            ax.axhline(reference, color=COLOR_UNKNOWN, linewidth=1.2,  # type: ignore[attr-defined]
                       linestyle="--", zorder=2)
            ax.annotate(  # type: ignore[attr-defined]
                "baseline mean", xy=(0.99, reference),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=8, color=TEXT_SECONDARY,
            )
    _style_axis(ax, title, "stride id", ylabel)


__all__ = ["pca_projection", "plot_results", "state_style"]
