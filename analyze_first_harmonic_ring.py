#!/usr/bin/env python3
"""Integrate the first-harmonic ring of density FFT snapshots at the (x, y) peak time."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class HarmonicResult:
    psi_path: Path
    p0: float | None
    peak_time: float
    peak_density: float
    ring_integral: float
    peak_radius: float
    inner_radius: float
    outer_radius: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each psi*.out file, find the time of maximum density at the requested (x, y) "
            "location, FFT the full 2D density snapshot at that time, and integrate the first-harmonic "
            "ring in k-space. The integrated magnitude is plotted against p0."
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
            "Save the p0-vs-ring plot instead of only showing it. "
            "Provide a path or leave empty to auto-name within the output folder."
        ),
    )
    parser.add_argument(
        "-show",
        dest="show",
        action="store_true",
        help="Force showing the plot even if saving.",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Disable interactive display (default is to show unless saving).",
    )
    parser.set_defaults(show=None)
    return parser.parse_args()


def read_grid_parameters(sim_name: str, input_root: str) -> tuple[int, float]:
    input_path = Path(input_root) / sim_name / "input.in"
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find {input_path}")

    data = np.genfromtxt(str(input_path), skip_footer=1, comments="!")
    if data.size < 10:
        raise ValueError(f"Input file {input_path} does not contain enough entries.")

    nodes_per_dim = int(data[0])
    num_crit = float(data[8])
    return nodes_per_dim, num_crit


def compute_grid(nodes_per_dim: int, num_crit: float) -> tuple[np.ndarray, float]:
    l_dom = 2.0 * np.pi * num_crit
    hx = l_dom / float(nodes_per_dim)
    grid = np.linspace(0.0, l_dom - hx, nodes_per_dim) - l_dom / 2.0
    return grid, l_dom


def resolve_psi_files(sim_name: str, output_root: str, index: int | None) -> list[Path]:
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
        raise ValueError(f"Unexpected number of columns ({num_grid_points}) in {psi_path}.")
    density_cube = density_values.reshape(len(times), nodes, nodes)
    return times, density_cube


def find_nearest(value_array: np.ndarray, target: float) -> tuple[int, float]:
    idx = int(np.argmin(np.abs(value_array - target)))
    return idx, float(value_array[idx])


def compute_fft2(density_plane: np.ndarray, l_dom: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = density_plane.shape[0]
    hx = l_dom / float(nodes)
    rho = density_plane / np.mean(density_plane)
    fft_vals = np.fft.fftshift(np.fft.fft2(rho))
    norm_factor = (hx * hx) / (l_dom * l_dom)
    fft_vals = norm_factor * fft_vals

    k_vals = np.fft.fftshift(np.fft.fftfreq(nodes, d=hx) * 2.0 * np.pi)
    return k_vals, k_vals, fft_vals


def suppress_central_peak(fft_vals: np.ndarray, radius: int = 1) -> np.ndarray:
    cleaned = fft_vals.copy()
    center = np.array(cleaned.shape) // 2
    r = max(0, radius)
    y0 = max(center[0] - r, 0)
    y1 = min(center[0] + r + 1, cleaned.shape[0])
    x0 = max(center[1] - r, 0)
    x1 = min(center[1] + r + 1, cleaned.shape[1])
    cleaned[y0:y1, x0:x1] = 0.0
    return cleaned


def detect_first_harmonic_ring(
    fft_mag: np.ndarray,
    kx_vals: np.ndarray,
    ky_vals: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    KX, KY = np.meshgrid(kx_vals, ky_vals, indexing="xy")
    radial_dist = np.sqrt(KX * KX + KY * KY)
    peak_idx = int(np.argmax(fft_mag))
    peak_radius = float(radial_dist.flat[peak_idx])
    peak_value = float(fft_mag.flat[peak_idx])

    if np.isclose(peak_radius, 0.0):
        # fallback: locate the largest non-central value manually
        masked = fft_mag.copy()
        center = np.array(masked.shape) // 2
        masked[center[0], center[1]] = 0.0
        peak_idx = int(np.argmax(masked))
        peak_radius = float(radial_dist.flat[peak_idx])
        peak_value = float(masked.flat[peak_idx])

    dkx = float(np.abs(kx_vals[1] - kx_vals[0])) if len(kx_vals) > 1 else 1.0
    dky = float(np.abs(ky_vals[1] - ky_vals[0])) if len(ky_vals) > 1 else 1.0
    radial_step = max(dkx, dky)
    radial_window = max(2.5 * radial_step, 0.05 * peak_radius) if peak_radius > 0 else 2.5 * radial_step
    if radial_window == 0.0:
        radial_window = 2.0 * radial_step if radial_step > 0 else 1.0

    amplitude_mask = fft_mag >= (0.5 * peak_value)
    ring_mask = np.abs(radial_dist - peak_radius) <= radial_window
    combined_mask = amplitude_mask & ring_mask
    candidate_radii = radial_dist[combined_mask]
    if candidate_radii.size == 0:
        candidate_radii = np.array([peak_radius])

    inner_radius = max(float(np.min(candidate_radii)) - radial_step, 0.0)
    outer_radius = float(np.max(candidate_radii)) + radial_step
    if not np.isfinite(inner_radius) or not np.isfinite(outer_radius):
        inner_radius = max(peak_radius - radial_step, 0.0)
        outer_radius = peak_radius + radial_step
    if outer_radius <= inner_radius:
        outer_radius = inner_radius + max(radial_step, 1.0)

    ring_mask = (radial_dist >= inner_radius) & (radial_dist <= outer_radius)
    return inner_radius, outer_radius, peak_radius, ring_mask


def integrate_ring(
    fft_mag: np.ndarray,
    ring_mask: np.ndarray,
    dkx: float,
    dky: float,
) -> float:
    if not np.any(ring_mask):
        return 0.0
    area_element = dkx * dky
    return float(np.sum(fft_mag[ring_mask]) * area_element)


def parse_p0_from_name(path: Path) -> float | None:
    stem = path.stem
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
    l_dom: float,
) -> HarmonicResult:
    times, density_cube = load_density_data(psi_path)
    nodes = density_cube.shape[1]
    if nodes != len(grid):
        raise ValueError(
            f"Grid size mismatch for {psi_path}: cube has {nodes}, grid has {len(grid)}."
        )

    density_trace = density_cube[:, y_idx, x_idx]
    peak_idx = int(np.argmax(density_trace))
    peak_time = float(times[peak_idx])
    peak_density = float(density_trace[peak_idx])

    snapshot = density_cube[peak_idx, :, :]
    kx_vals, ky_vals, fft_vals = compute_fft2(snapshot, l_dom)
    fft_vals = suppress_central_peak(fft_vals, radius=1)
    fft_mag = np.abs(fft_vals)

    inner_radius, outer_radius, peak_radius, ring_mask = detect_first_harmonic_ring(
        fft_mag, kx_vals, ky_vals
    )
    dkx = float(np.abs(kx_vals[1] - kx_vals[0])) if len(kx_vals) > 1 else 1.0
    dky = float(np.abs(ky_vals[1] - ky_vals[0])) if len(ky_vals) > 1 else 1.0
    ring_integral = integrate_ring(fft_mag, ring_mask, dkx, dky)

    return HarmonicResult(
        psi_path=psi_path,
        p0=parse_p0_from_name(psi_path),
        peak_time=peak_time,
        peak_density=peak_density,
        ring_integral=ring_integral,
        peak_radius=peak_radius,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )


def summarize_results(results: Sequence[HarmonicResult]) -> None:
    header = (
        f"{'psi file':<30}"
        f"{'p0':>15}"
        f"{'t_peak':>15}"
        f"{'density_peak':>18}"
        f"{'ring_integral':>18}"
        f"{'k_peak':>15}"
        f"{'k_inner':>15}"
        f"{'k_outer':>15}"
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
            f"{res.ring_integral:>18.6e}"
            f"{res.peak_radius:>15.6e}"
            f"{res.inner_radius:>15.6e}"
            f"{res.outer_radius:>15.6e}"
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
    return first_psi_path.with_name("first_harmonic_ring_integrals.png")


def plot_results(
    results: Sequence[HarmonicResult],
    save_requested: bool,
    save_arg: str | None,
    show_pref: bool,
) -> None:
    if not results:
        print("No results to plot.")
        return

    plt = get_matplotlib()
    have_p0 = all(res.p0 is not None for res in results)

    if have_p0:
        order = np.argsort([res.p0 for res in results])
        xs = np.array([results[i].p0 for i in order], dtype=float)
        ys = np.array([results[i].ring_integral for i in order], dtype=float)
        x_label = r"$p_0$"
    else:
        xs = np.arange(len(results))
        ys = np.array([res.ring_integral for res in results], dtype=float)
        x_label = "psi file index"

    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Integrated first-harmonic magnitude")
    ax.set_title("First-harmonic ring integral vs p0")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.1)

    fig.tight_layout()

    if save_requested:
        first_path = results[0].psi_path
        save_path = (
            Path(save_arg)
            if save_arg
            else build_default_save_path(first_path)
        )
        fig.savefig(save_path, dpi=200)
        print(f"Saved p0-vs-ring plot to {save_path}")

    should_show = show_pref or not save_requested
    if should_show:
        plt.show()

    plt.close(fig)


def main() -> None:
    args = parse_args()
    save_requested = args.save is not None
    show_pref = True if args.show is None else args.show

    nodes_per_dim, num_crit = read_grid_parameters(args.filename, args.input_root)
    grid, l_dom = compute_grid(nodes_per_dim, num_crit)

    x_idx, x_selected = find_nearest(grid, args.x)
    y_idx, y_selected = find_nearest(grid, args.y)

    psi_files = resolve_psi_files(args.filename, args.output_root, args.index)

    results: list[HarmonicResult] = []
    for psi_path in psi_files:
        result = analyze_file(psi_path, x_idx, y_idx, grid, l_dom)
        results.append(result)

    summarize_results(results)
    print()
    print(
        f"Integrated first-harmonic ring at t_peak for x={x_selected:.4f}, "
        f"y={y_selected:.4f} over {len(results)} psi file(s)."
    )

    plot_results(
        results,
        save_requested=save_requested,
        save_arg=args.save if isinstance(args.save, str) else None,
        show_pref=show_pref,
    )


if __name__ == "__main__":
    main()
