#!/usr/bin/env python3
"""Take a spatial density slice at a chosen time, FFT it, and plot both."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the density snapshot nearest to the requested time, identify its peak, "
            "plot the 2D density, and plot its 2D Fourier transform as a contour map (no 3D surface)."
        )
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Simulation/output folder name (matches patt2d_q_sfm --filename).",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        required=True,
        help="Target simulation time for the snapshot.",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        help="Optional frame index for multi-frame runs (matches patt2d_q_sfm --index).",
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
            "Save the figures instead of showing them. Provide a path prefix or leave empty "
            "to auto-name within the output folder; two files will be created with "
            "suffixes '_density' and '_fftcontour'."
        ),
    )
    parser.add_argument(
        "-show",
        dest="show",
        action="store_true",
        help="Force showing plots even if saving.",
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


def resolve_psi_file(sim_name: str, output_root: str, index: int | None) -> Path:
    sim_dir = Path(output_root) / sim_name
    if not sim_dir.exists():
        raise FileNotFoundError(f"Output directory {sim_dir} does not exist.")

    if index is None:
        default_file = sim_dir / "psi.out"
        if default_file.exists():
            return default_file
        psi_files = sorted(sim_dir.glob("psi*.out"))
        if len(psi_files) == 1:
            return psi_files[0]
        raise FileNotFoundError(
            "Could not determine which psi file to open. Specify --index to disambiguate."
        )

    pattern_with_param = sim_dir.glob(f"psi{index}_*.out")
    matches = sorted(pattern_with_param)
    if not matches:
        legacy_file = sim_dir / f"psi{index}.out"
        if legacy_file.exists():
            return legacy_file
        raise FileNotFoundError(f"No psi file found for index {index} in {sim_dir}.")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple psi files matched index {index}: "
            + ", ".join(str(p.name) for p in matches)
        )
    return matches[0]


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
    # Normalize density so <rho>=1 to mirror the 1D helper
    rho = density_plane / np.mean(density_plane)
    fft_vals = np.fft.fftshift(np.fft.fft2(rho))
    # Continuum-normalized coefficient akin to 1D: (dx dy / L^2) sum rho e^{-i(kx x + ky y)}
    norm_factor = (hx * hx) / (l_dom * l_dom)
    fft_vals = norm_factor * fft_vals

    k_vals = np.fft.fftshift(np.fft.fftfreq(nodes, d=hx) * 2.0 * np.pi)
    return k_vals, k_vals, fft_vals


def suppress_central_peak(fft_vals: np.ndarray, radius: int = 1) -> np.ndarray:
    """Zero-out the dc/central harmonic so it does not dominate the contour plot."""
    cleaned = fft_vals.copy()
    center = np.array(cleaned.shape) // 2
    r = max(0, radius)
    y0 = max(center[0] - r, 0)
    y1 = min(center[0] + r + 1, cleaned.shape[0])
    x0 = max(center[1] - r, 0)
    x1 = min(center[1] + r + 1, cleaned.shape[1])
    cleaned[y0:y1, x0:x1] = 0.0
    return cleaned


def get_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with "
            "`pip install matplotlib` and re-run the script."
        ) from exc
    return plt


def build_save_paths(base_path: Path, time_val: float) -> tuple[Path, Path]:
    def sanitize(val: float) -> str:
        return f"{val:+.3f}".replace("-", "m").replace("+", "p").replace(".", "p")

    suffix = sanitize(time_val)
    density_path = base_path.with_name(f"{base_path.stem}_density_t{suffix}.png")
    fft_contour_path = base_path.with_name(f"{base_path.stem}_fftcontour_t{suffix}.png")
    return density_path, fft_contour_path


def main() -> None:
    args = parse_args()

    save_requested = args.save is not None
    show_pref = True if args.show is None else args.show

    plt = get_matplotlib()
    from matplotlib.patches import Wedge  # type: ignore

    nodes_per_dim, num_crit = read_grid_parameters(args.filename, args.input_root)
    grid, l_dom = compute_grid(nodes_per_dim, num_crit)

    psi_path = resolve_psi_file(args.filename, args.output_root, args.index)
    times, density_cube = load_density_data(psi_path)
    if density_cube.shape[1] != nodes_per_dim:
        raise ValueError("Grid size inferred from output does not match input file settings.")

    time_idx, time_selected = find_nearest(times, args.time)
    density_plane = density_cube[time_idx, :, :]

    max_idx_flat = int(np.argmax(density_plane))
    max_y_idx, max_x_idx = np.unravel_index(max_idx_flat, density_plane.shape)
    max_value = float(density_plane[max_y_idx, max_x_idx])
    max_x = float(grid[max_x_idx])
    max_y = float(grid[max_y_idx])

    kx_vals, ky_vals, fft_vals = compute_fft2(density_plane, l_dom)
    fft_vals = suppress_central_peak(fft_vals, radius=1)
    KX, KY = np.meshgrid(kx_vals, ky_vals, indexing="xy")
    fft_mag = np.abs(fft_vals)
    radial_dist = np.sqrt(KX * KX + KY * KY)
    peak_flat_idx = int(np.argmax(fft_mag))
    peak_radius = float(radial_dist.flat[peak_flat_idx])
    k_spacing = float(np.abs(kx_vals[1] - kx_vals[0])) if len(kx_vals) > 1 else 0.0
    max_fft_val = float(fft_mag.flat[peak_flat_idx])
    amplitude_mask = fft_mag >= (0.5 * max_fft_val)
    radial_window = max(2.5 * k_spacing, 0.05 * peak_radius) if peak_radius > 0 else max(2.5 * k_spacing, 0.0)
    if radial_window == 0.0:
        radial_window = 2.0 * (k_spacing if k_spacing > 0 else 1.0)
    candidate_mask = np.abs(radial_dist - peak_radius) <= radial_window
    peak_band_mask = candidate_mask & amplitude_mask
    candidate_radii = radial_dist[peak_band_mask]
    if candidate_radii.size == 0:
        candidate_radii = np.array([peak_radius])
    inner_radius = max(float(np.min(candidate_radii)) - k_spacing, 0.0)
    outer_radius = float(np.max(candidate_radii)) + k_spacing
    if not np.isfinite(inner_radius) or not np.isfinite(outer_radius):
        inner_radius = max(peak_radius - k_spacing, 0.0)
        outer_radius = peak_radius + k_spacing
    if outer_radius <= inner_radius:
        pad = k_spacing if k_spacing > 0 else max(0.05 * peak_radius, 1.0)
        outer_radius = inner_radius + pad
    ring_width = outer_radius - inner_radius

    fig1, ax1 = plt.subplots()
    im = ax1.imshow(
        density_plane,
        extent=[grid.min(), grid.max(), grid.min(), grid.max()],
        origin="lower",
        cmap="viridis",
    )
    ax1.set_aspect("equal", adjustable="box")
    ax1.scatter(max_x, max_y, color="red", s=30, label=f"max={max_value:.3e}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(rf"Density at t={time_selected:.3e} ({args.filename})")
    ax1.legend(loc="upper right")
    fig1.colorbar(im, ax=ax1, label=r"$|\psi|^2$")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots()
    cont = ax2.contourf(KX, KY, fft_mag, levels=40, cmap="plasma")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel(r"$k_x$")
    ax2.set_ylabel(r"$k_y$")
    ax2.set_title(r"FFT magnitude of density (contour)")
    ring = Wedge(
        (0.0, 0.0),
        outer_radius,
        0.0,
        360.0,
        width=ring_width,
        facecolor="none",
        edgecolor="w",
        linewidth=1.8,
        linestyle="--",
        alpha=0.9,
    )
    ax2.add_patch(ring)
    fig2.colorbar(cont, ax=ax2, label=r"$|n(k_x, k_y)|$")
    fig2.tight_layout()

    save_path_density: Path | None = None
    save_path_fftcontour: Path | None = None
    if save_requested:
        base = Path(args.save) if args.save else psi_path
        save_path_density, save_path_fftcontour = build_save_paths(base, time_selected)
        fig1.savefig(save_path_density, dpi=200)
        fig2.savefig(save_path_fftcontour, dpi=200)
        print(f"Saved density plot to {save_path_density}")
        print(f"Saved FFT contour plot to {save_path_fftcontour}")

    print(
        f"Using snapshot at t={time_selected:.4e} (requested {args.time}), "
        f"peak density {max_value:.3e} at x={max_x:.4f}, y={max_y:.4f}."
    )

    should_show = show_pref or not save_requested
    if should_show:
        plt.show()

    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    main()
