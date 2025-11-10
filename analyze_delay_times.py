#!/usr/bin/env python3
"""Estimate delay times from simulation outputs and compare to analytic predictions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

try:
    from analytic_predictors import analytic_delay_time, pump_threshold
except ImportError as exc:
    raise SystemExit(
        "Could not import analytic predictors. Make sure analytic_predictors.py "
        "is on the Python path."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the first real density peak at a specified (x, y) location for every "
            "p0 inside a simulation folder, and compare with analytic delay times."
        )
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Simulation/output folder name (matches patt2d_q_sfm --filename).",
    )
    parser.add_argument("-x", type=float, required=True, help="Target x coordinate.")
    parser.add_argument("-y", type=float, required=True, help="Target y coordinate.")
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
        "--min-prominence",
        type=float,
        default=None,
        help="Optional prominence threshold for scipy.signal.find_peaks.",
    )
    parser.add_argument(
        "--save",
        metavar="path",
        help="Optional path to save the delay-time plot (PNG).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable the interactive plot window.",
    )
    return parser.parse_args()


def read_input_parameters(sim_name: str, input_root: str) -> dict[str, float]:
    input_path = Path(input_root) / sim_name / "input.in"
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find {input_path}")

    data = np.genfromtxt(str(input_path), skip_footer=1, comments="!")
    if data.size not in (11, 12):
        raise ValueError(
            f"Input file {input_path} must contain 11 or 12 numeric entries (excluding flags)."
        )

    params = {
        "nodes_per_dim": int(data[0]),
        "maxt": float(data[1]),
        "ht": float(data[2]),
        "width_psi": float(data[3]),
        "p0": float(data[4]),
        "Delta": float(data[5]),
        "omega_r": float(data[6]),
        "b0": float(data[7]),
        "num_crit": float(data[8]),
        "R": float(data[9]),
        "plotnum": int(data[10]),
        "seed_amp": float(data[11]) if data.size >= 12 else 1.0e-6,
    }
    return params


def build_grid(nodes_per_dim: int, num_crit: float) -> np.ndarray:
    l_dom = 2.0 * np.pi * num_crit
    hx = l_dom / float(nodes_per_dim)
    return np.linspace(0.0, l_dom - hx, nodes_per_dim) - l_dom / 2.0


def load_density_cube(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(str(path))
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    times = raw[:, 0]
    values = raw[:, 1:]
    grid_points = values.shape[1]
    nodes = int(round(np.sqrt(grid_points)))
    if nodes * nodes != grid_points:
        raise ValueError(f"{path} does not contain a square grid (columns={grid_points}).")
    cube = values.reshape(len(times), nodes, nodes)
    return times, cube


def nearest_index(grid: np.ndarray, coord: float) -> tuple[int, float]:
    idx = int(np.argmin(np.abs(grid - coord)))
    return idx, float(grid[idx])


def extract_p0_from_name(path: Path) -> float | None:
    match = re.search(r"psi\d*_(?P<pval>[-+0-9.eE]+)$", path.stem)
    if not match:
        return None
    try:
        return float(match.group("pval"))
    except ValueError:
        return None


def resolve_peak_finder():
    try:
        from scipy.signal import find_peaks as scipy_find_peaks  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit("scipy is required. Install it with `pip install scipy`.") from exc
    return scipy_find_peaks


def main() -> None:
    args = parse_args()
    find_peaks = resolve_peak_finder()

    params = read_input_parameters(args.filename, args.input_root)
    grid = build_grid(params["nodes_per_dim"], params["num_crit"])
    x_idx, x_val = nearest_index(grid, args.x)
    y_idx, y_val = nearest_index(grid, args.y)

    sim_dir = Path(args.output_root) / args.filename
    psi_files = sorted(sim_dir.glob("psi*.out"))
    if not psi_files:
        raise FileNotFoundError(f"No psi*.out files found inside {sim_dir}")

    p_thresh = pump_threshold(params["omega_r"], params["b0"], params["R"])
    if not np.isfinite(p_thresh):
        raise RuntimeError("Analytic pump threshold is not finite; cannot proceed.")
    seed_amp = params["seed_amp"]

    sim_p_values: list[float] = []
    sim_delay_times: list[float] = []
    analytic_p_values: list[float] = []
    analytic_delay_times: list[float] = []

    for psi_path in psi_files:
        p0_val = extract_p0_from_name(psi_path)
        if p0_val is None:
            print(f"Skipping {psi_path.name}: could not extract p0 from filename.")
            continue

        if p0_val <= p_thresh:
            print(
                f"Skipping {psi_path.name}: p0={p0_val:.3e} <= p_th={p_thresh:.3e}."
            )
            continue

        times, density_cube = load_density_cube(psi_path)
        if density_cube.shape[1] != params["nodes_per_dim"]:
            raise ValueError(
                f"Grid mismatch for {psi_path.name}: expected {params['nodes_per_dim']} nodes."
            )

        if len(times) < 3:
            print(f"{psi_path.name}: not enough time samples for peak detection.")
            continue

        trace = density_cube[:, y_idx, x_idx]
        peaks, _ = find_peaks(trace, prominence=args.min_prominence)

        if peaks.size == 0:
            print(f"{psi_path.name}: no real peaks detected at x={x_val:.3f}, y={y_val:.3f}.")
            analytic_val = analytic_delay_time(
                p0_val, p_thresh, params["omega_r"], seed_amp
            )
            if np.isfinite(analytic_val):
                analytic_p_values.append(p0_val)
                analytic_delay_times.append(float(analytic_val))
                print(
                    f"{psi_path.name}: analytic delay={analytic_val:.3e} (no simulated peak)."
                )
            continue

        first_peak_idx = int(peaks[0])
        delay_time = float(times[first_peak_idx])

        analytic_time = analytic_delay_time(p0_val, p_thresh, params["omega_r"], seed_amp)

        if np.isfinite(analytic_time):
            analytic_p_values.append(p0_val)
            analytic_delay_times.append(float(analytic_time))
        else:
            print(f"{psi_path.name}: analytic delay time is not finite.")

        sim_p_values.append(p0_val)
        sim_delay_times.append(delay_time)

        if np.isfinite(analytic_time):
            print(
                f"{psi_path.name}: p0={p0_val:.3e}, delay={delay_time:.3e}, "
                f"analytic={analytic_time:.3e}"
            )
        else:
            print(f"{psi_path.name}: p0={p0_val:.3e}, delay={delay_time:.3e}")

    if not sim_p_values:
        print("No valid delay times were found.")
        return

    order_sim = np.argsort(sim_p_values)
    sim_p_arr = np.array(sim_p_values)[order_sim]
    sim_delay_arr = np.array(sim_delay_times)[order_sim]

    if analytic_p_values:
        order_analytic = np.argsort(analytic_p_values)
        analytic_p_arr = np.array(analytic_p_values)[order_analytic]
        analytic_arr = np.array(analytic_delay_times)[order_analytic]
    else:
        analytic_p_arr = np.array([])
        analytic_arr = np.array([])

    import matplotlib.pyplot as plt  # pylint: disable=import-error

    fig, ax = plt.subplots()
    ax.scatter(sim_p_arr, sim_delay_arr, label="Simulation delay", color="tab:blue")
    if analytic_p_arr.size:
        ax.plot(
            analytic_p_arr,
            analytic_arr,
            label="Analytic delay",
            color="tab:orange",
            linestyle="--",
        )
    ax.set_xlabel("p0")
    ax.set_ylabel("Delay time")
    ax.set_title(
        rf"Delay time at x={x_val:.3f}, y={y_val:.3f} for {args.filename} "
        r"(p0 > $p_{\mathrm{th}}$)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=200)
        print(f"Saved plot to {args.save}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
