#!/usr/bin/env python3
"""Plot a 1D density cut at a specific simulation time."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a density cut (along x) at a specific simulation time."
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Simulation/output folder name (matches patt2d_q_sfm --filename).",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        help=(
            "Optional frame index for multi-frame runs (matches patt2d_q_sfm --index). "
            "When omitted a unique psi*.out file inside the run folder is used."
        ),
    )
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        required=True,
        help="Target simulation time for the cut.",
    )
    parser.add_argument(
        "-y",
        "--y-coordinate",
        type=float,
        default=0.0,
        help="Y coordinate for the cut (default: 0, uses nearest grid point).",
    )
    parser.add_argument(
        "--input-root",
        default="inputs",
        help="Root directory where simulation input folders live (default: inputs).",
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
            "Save the figure instead of showing it. Provide a path or leave empty "
            "to auto-name within the output folder."
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


def compute_grid(nodes_per_dim: int, num_crit: float) -> np.ndarray:
    l_dom = 2.0 * np.pi * num_crit
    hx = l_dom / float(nodes_per_dim)
    return np.linspace(0.0, l_dom - hx, nodes_per_dim) - l_dom / 2.0


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
            "Could not determine which psi file to open. "
            "Specify --index to disambiguate."
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


def build_default_save_path(psi_path: Path, time_val: float, y_coord: float) -> Path:
    def sanitize(val: float) -> str:
        return f"{val:+.3f}".replace("-", "m").replace("+", "p").replace(".", "p")

    suffix = f"cut_t{sanitize(time_val)}_y{sanitize(y_coord)}.png"
    return psi_path.with_name(f"{psi_path.stem}_{suffix}")


def get_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with "
            "`pip install matplotlib` and re-run the script."
        ) from exc
    return plt


def main() -> None:
    args = parse_args()

    save_requested = args.save is not None
    show_pref = True if args.show is None else args.show

    plt = get_matplotlib()

    nodes_per_dim, num_crit = read_grid_parameters(args.filename, args.input_root)
    grid = compute_grid(nodes_per_dim, num_crit)

    psi_path = resolve_psi_file(args.filename, args.output_root, args.index)

    times, density_cube = load_density_data(psi_path)
    if density_cube.shape[1] != nodes_per_dim:
        raise ValueError(
            "Grid size inferred from output does not match input file settings."
        )

    time_idx, time_selected = find_nearest(times, args.time)
    y_idx, y_selected = find_nearest(grid, args.y_coordinate)

    density_slice = density_cube[time_idx, y_idx, :]

    fig, ax = plt.subplots()
    ax.plot(grid, density_slice)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\psi(x, y, t)|^2$")
    ax.set_title(
        rf"Density cut at t={time_selected:.3e}, y={y_selected:.3f} ({args.filename})"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path: Path | None = None
    if save_requested:
        save_path = (
            Path(args.save)
            if args.save
            else build_default_save_path(psi_path, time_selected, y_selected)
        )
        fig.savefig(save_path, dpi=200)

    print(
        f"Extracted density cut at t={time_selected:.4e} "
        f"(requested {args.time}) from {psi_path.name}."
    )
    print(f"Using y={y_selected:.4f} (nearest grid point).")

    if save_requested:
        print(f"Saved plot to {save_path}")
        if args.show is False:
            print("Interactive display suppressed because --save was requested.")

    should_show = show_pref and not save_requested

    if should_show:
        print("Displaying interactive plot (close the window to exit).")
        plt.show()
        plt.close(fig)
    else:
        if not save_requested:
            print("Plot not saved or shown. Use --save to export.")
        plt.close(fig)


if __name__ == "__main__":
    main()
