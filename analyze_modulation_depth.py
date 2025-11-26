#!/usr/bin/env python3
"""Compute modulation depth from full 2D psi density snapshots at peak times."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import math
try:
    from analytic_predictors import pump_threshold
except ImportError as exc:
    raise SystemExit(
        "Could not import analytic predictors. Ensure analytic_predictors.py is available."
    ) from exc


@dataclass
class ModulationResult:
    psi_path: Path
    p0: float | None
    peak_time: float
    peak_density: float
    modulation_depth: float
    plane_max: float
    plane_min: float


@dataclass
class SimulationParameters:
    nodes_per_dim: int
    num_crit: float
    omega_r: float
    b0: float
    reflectivity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locate the time of maximum density at a (x, y) point, then evaluate "
            "the modulation depth using the full 2D density snapshot at that time for each psi file."
        )
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Simulation name (same as patt2d_q_sfm --filename).",
    )
    parser.add_argument("-x", type=float, required=True, help="Target x coordinate.")
    parser.add_argument("-y", type=float, required=True, help="Target y coordinate.")
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        help="Optional psi index to restrict the analysis to a single file.",
    )
    parser.add_argument(
        "--input-root",
        default="inputs",
        help="Root directory where simulation inputs live (default: inputs).",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root directory where simulation outputs live (default: outputs).",
    )
    parser.add_argument(
        "-save",
        nargs="?",
        const="",
        metavar="path",
        help=(
            "Save the modulation-depth plot instead of only printing values. "
            "Provide a path or leave empty to auto-name within the output folder."
        ),
    )
    parser.add_argument(
        "-show",
        dest="show",
        action="store_true",
        help="Force showing the plot even if disabled elsewhere.",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Disable interactive display (default is to show).",
    )
    parser.set_defaults(show=None)
    return parser.parse_args()


def read_simulation_parameters(sim_name: str, input_root: str) -> SimulationParameters:
    input_path = Path(input_root) / sim_name / "input.in"
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find {input_path}")

    data = np.genfromtxt(str(input_path), skip_footer=1, comments="!")
    if data.size < 10:
        raise ValueError(
            f"Input file {input_path} does not contain enough entries "
            "(at least 10 numeric values expected)."
        )

    return SimulationParameters(
        nodes_per_dim=int(data[0]),
        num_crit=float(data[8]),
        omega_r=float(data[6]),
        b0=float(data[7]),
        reflectivity=float(data[9]),
    )


def compute_grid(nodes_per_dim: int, num_crit: float) -> np.ndarray:
    l_dom = 2.0 * np.pi * num_crit
    hx = l_dom / float(nodes_per_dim)
    return np.linspace(0.0, l_dom - hx, nodes_per_dim) - l_dom / 2.0


def resolve_psi_files(
    sim_name: str, output_root: str, index: int | None
) -> list[Path]:
    sim_dir = Path(output_root) / sim_name
    if not sim_dir.exists():
        raise FileNotFoundError(f"Output directory {sim_dir} does not exist.")

    all_files = sorted(sim_dir.glob("psi*.out"))
    if not all_files:
        raise FileNotFoundError(f"No psi*.out files found inside {sim_dir}.")

    if index is None:
        return all_files

    matched: list[Path] = []
    pattern_with_param = sim_dir.glob(f"psi{index}_*.out")
    matched.extend(pattern_with_param)
    legacy_file = sim_dir / f"psi{index}.out"
    if legacy_file.exists():
        matched.append(legacy_file)

    if not matched:
        raise FileNotFoundError(
            f"No psi file found for index {index} in directory {sim_dir}."
        )
    return sorted(matched)


def load_density_data(psi_path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(str(psi_path))
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    times = raw[:, 0]
    density_values = raw[:, 1:]
    num_grid_points = density_values.shape[1]
    nodes = int(round(np.sqrt(num_grid_points)))
    if nodes * nodes != num_grid_points:
        raise ValueError(
            f"Unexpected number of columns ({num_grid_points}) in {psi_path}."
        )
    density_cube = density_values.reshape(len(times), nodes, nodes)
    return times, density_cube


def find_nearest(value_array: np.ndarray, target: float) -> tuple[int, float]:
    idx = int(np.argmin(np.abs(value_array - target)))
    return idx, float(value_array[idx])


def parse_p0_from_name(path: Path) -> float | None:
    stem = path.stem  # e.g., psi0_1.98e-10
    if "_" not in stem:
        return None
    suffix = stem.split("_", 1)[1]
    try:
        return float(suffix)
    except ValueError:
        return None


def analyze_file(
    psi_path: Path,
    x_idx: int,
    y_idx: int,
    grid: np.ndarray,
) -> ModulationResult:
    times, density_cube = load_density_data(psi_path)
    nodes = density_cube.shape[1]
    if nodes != len(grid):
        raise ValueError(
            "Grid size inferred from psi output does not match the input file settings."
        )

    density_trace = density_cube[:, y_idx, x_idx]
    peak_idx = int(np.argmax(density_trace))
    peak_time = float(times[peak_idx])
    peak_density = float(density_trace[peak_idx])

    snapshot = density_cube[peak_idx, :, :]
    plane_max = float(np.max(snapshot))
    plane_min = float(np.min(snapshot))
    denominator = plane_max + plane_min
    if np.isclose(denominator, 0.0):
        modulation_depth = np.nan
    else:
        modulation_depth = (plane_max - plane_min) / denominator

    return ModulationResult(
        psi_path=psi_path,
        p0=parse_p0_from_name(psi_path),
        peak_time=peak_time,
        peak_density=peak_density,
        modulation_depth=float(modulation_depth),
        plane_max=plane_max,
        plane_min=plane_min,
    )


def summarize_results(results: Sequence[ModulationResult]) -> None:
    header = (
        f"{'psi file':<30}"
        f"{'p0':>15}"
        f"{'t_peak':>15}"
        f"{'density_peak':>18}"
        f"{'mod_depth':>15}"
        f"{'plane_max':>15}"
        f"{'plane_min':>15}"
    )
    print(header)
    print("-" * len(header))
    for res in results:
        p0_str = f"{res.p0:.6g}" if res.p0 is not None else "n/a"
        print(
            f"{res.psi_path.name:<30}"
            f"{p0_str:>15}"
            f"{res.peak_time:>15.6e}"
            f"{res.peak_density:>18.6e}"
            f"{res.modulation_depth:>15.6e}"
            f"{res.plane_max:>15.6e}"
            f"{res.plane_min:>15.6e}"
        )


def get_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with "
            "`pip install matplotlib` and re-run the script."
        ) from exc
    return plt


def build_default_save_path(first_psi_path: Path) -> Path:
    return first_psi_path.with_name("modulation_depths.png")


def plot_modulation_depths(
    results: Sequence[ModulationResult],
    save_requested: bool,
    save_arg: str | None,
    show_pref: bool,
    p_threshold: float | None,
) -> None:
    plt = get_matplotlib()

    # Local style tweaks for a clean, publication-ready look.
    style_overrides = {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#1f2933",
        "axes.linewidth": 0.9,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.color": "#d8dee9",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
    }

    with plt.rc_context(style_overrides):
        have_p0 = all(res.p0 is not None for res in results)

        if have_p0:
            order = np.argsort([res.p0 for res in results])
            xs = np.array([results[i].p0 for i in order], dtype=float)
            ys = np.array([results[i].modulation_depth for i in order], dtype=float)
            labels = [results[i].psi_path.name for i in order]
            x_label = r"$p_0$"
        else:
            xs = np.arange(len(results))
            ys = np.array([res.modulation_depth for res in results], dtype=float)
            labels = [res.psi_path.name for res in results]
            x_label = "psi file index"

        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        line_color = "#1d4f91"
        marker_edge = "#0b3954"
        sim_line, = ax.plot(
            xs,
            ys,
            color=line_color,
            linewidth=2.0,
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=marker_edge,
            markeredgewidth=1.4,
            label="Simulation",
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Modulation depth")
        ax.set_title("Modulation depth at peak density")
        ax.set_ylim(0.0, 1.05)
        ax.margins(x=0.04)
        ax.grid(True, axis="both", alpha=0.55)

        # Add analytical threshold marker when available and relevant.
        if have_p0 and p_threshold is not None and math.isfinite(p_threshold):
            threshold_color = "#c43e31"
            ax.axvline(
                p_threshold,
                color=threshold_color,
                linestyle="--",
                linewidth=1.4,
                label=r"$p_{th}$",
                alpha=0.9,
            )
            y_top = ax.get_ylim()[1]
            ax.annotate(
                r"$p_{th}$",
                xy=(p_threshold, y_top),
                xytext=(0, -12),
                textcoords="offset points",
                color=threshold_color,
                ha="center",
                va="top",
                fontsize=10,
            )

        # Annotate each point with the corresponding psi file stem when p0 is missing.
        if not have_p0:
            for x_val, y_val, label in zip(xs, ys, labels):
                if math.isfinite(y_val):
                    ax.annotate(
                        label,
                        (x_val, y_val),
                        textcoords="offset points",
                        xytext=(0, 7),
                        ha="center",
                        fontsize=8,
                        color="#444444",
                    )

        # Minimalist legend when threshold is shown.
        if have_p0 and p_threshold is not None and math.isfinite(p_threshold):
            ax.legend(frameon=False, loc="upper left")

        fig.tight_layout()

        save_path: Path | None = None
        if save_requested:
            first_path = results[0].psi_path
            save_path = Path(save_arg) if save_arg else build_default_save_path(first_path)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved modulation-depth plot to {save_path}")

        should_show = show_pref and not save_requested
        if should_show:
            plt.show()

        plt.close(fig)


def main() -> None:
    args = parse_args()
    save_requested = args.save is not None
    show_pref = True if args.show is None else args.show

    params = read_simulation_parameters(args.filename, args.input_root)
    grid = compute_grid(params.nodes_per_dim, params.num_crit)
    p_threshold = pump_threshold(params.omega_r, params.b0, params.reflectivity)

    x_idx, x_selected = find_nearest(grid, args.x)
    y_idx, y_selected = find_nearest(grid, args.y)

    psi_files = resolve_psi_files(args.filename, args.output_root, args.index)

    results: list[ModulationResult] = []
    for psi_path in psi_files:
        result = analyze_file(psi_path, x_idx, y_idx, grid)
        results.append(result)

    summarize_results(results)

    print()
    print(
        f"Computed modulation depth using the full 2D snapshot at t_peak for "
        f"x={x_selected:.4f}, y={y_selected:.4f} over {len(results)} psi file(s)."
    )

    if results:
        plot_modulation_depths(
            results,
            save_requested=save_requested,
            save_arg=args.save if isinstance(args.save, str) else None,
            show_pref=show_pref,
            p_threshold=p_threshold,
        )
    else:
        print("No results to plot.")


if __name__ == "__main__":
    main()
