#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute modulation depth versus pump parameter by sampling the density
at a specified (x, y) coordinate and locating the time slice of peak amplitude.
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "For each pump frame produced by patt2d_q_sfm.py, "
            "find the maximum |Psi|^2 at a chosen (x,y) coordinate, "
            "take the full 2D slice at that time, compute its modulation depth, "
            "and plot the depth versus p0."
        )
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Simulation name (matches inputs/<name> and outputs/<name>).",
    )
    parser.add_argument(
        "--x",
        type=float,
        required=True,
        help="x coordinate (in the same units as the simulation domain).",
    )
    parser.add_argument(
        "--y",
        type=float,
        required=True,
        help="y coordinate (in the same units as the simulation domain).",
    )
    parser.add_argument(
        "--input-root",
        default="inputs",
        help="Root directory containing simulation inputs (default: inputs).",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root directory containing simulation outputs (default: outputs).",
    )
    parser.add_argument(
        "--save-fig",
        default="modulation_vs_p0.png",
        help="Filename for the scatter plot (saved inside the simulation's output folder).",
    )
    return parser.parse_args()


def read_input_metadata(input_file: Path):
    data = np.genfromtxt(str(input_file), skip_footer=1, comments="!")
    if data.size < 11:
        raise ValueError(f"Input file {input_file} is missing required entries.")

    nodes = int(data[0])
    num_crit = float(data[8])
    return nodes, num_crit


def compute_indices(x_coord, y_coord, nodes, num_crit):
    pi = np.pi
    domain_min = -pi * num_crit
    domain_max = pi * num_crit
    length = domain_max - domain_min
    hx = length / nodes

    def index_for(coord):
        idx = int(round((coord - domain_min) / hx))
        return max(0, min(nodes - 1, idx))

    return index_for(x_coord), index_for(y_coord)


def find_output_pairs(output_dir: Path):
    pairs = []
    # Handle base files first
    base_s = output_dir / "s.out"
    base_psi = output_dir / "psi.out"
    if base_s.exists() and base_psi.exists():
        pairs.append((base_s, base_psi, None))

    pattern = re.compile(r"s(\d+)_([0-9eE+\-\.]+)\.out$")
    for s_path in sorted(output_dir.glob("s*_*.out")):
        match = pattern.match(s_path.name)
        if not match:
            continue
        index = int(match.group(1))
        p0_value = float(match.group(2))
        psi_candidate = output_dir / s_path.name.replace("s", "psi", 1)
        if psi_candidate.exists():
            pairs.append((s_path, psi_candidate, p0_value))
        else:
            raise FileNotFoundError(
                f"Missing matching psi file for {s_path.name} in {output_dir}"
            )

    if not pairs:
        raise FileNotFoundError(f"No output files found in {output_dir}")

    return pairs


def modulation_depth(slice_2d: np.ndarray):
    return float(np.max(slice_2d) - np.min(slice_2d)) / (float(np.max(slice_2d) + np.min(slice_2d)))


def analyze_pair(
    s_path: Path,
    psi_path: Path,
    p0_value,
    nodes,
    coord_indices,
):
    psi_data = np.loadtxt(str(psi_path))
    times = psi_data[:, 0]
    psi_values = psi_data[:, 1:]

    xi, yi = coord_indices
    flat_index = yi * nodes + xi
    column = psi_values[:, flat_index]

    peak_idx = int(np.argmax(column))
    peak_time = times[peak_idx]

    slice_flat = psi_values[peak_idx]
    slice_2d = slice_flat.reshape(nodes, nodes)
    mod_depth = modulation_depth(slice_2d)

    inferred_p0 = p0_value
    if inferred_p0 is None:
        # No p0 encoded in filename; fallback to NaN so it stands out.
        inferred_p0 = np.nan

    return inferred_p0, mod_depth, peak_time


def plot_results(results, output_dir, save_name, x_coord, y_coord):
    results = [r for r in results if not np.isnan(r[0])]
    if not results:
        raise RuntimeError(
            "Could not infer any p0 values from the filenames. "
            "Ensure files are named like s<idx>_<p0>.out."
        )

    results.sort(key=lambda r: r[0])
    p0_vals = [r[0] for r in results]
    mod_depths = [r[1] for r in results]

    plt.figure(figsize=(6, 4))
    plt.plot(p0_vals, mod_depths, "o-")
    plt.xlabel("p0")
    plt.ylabel("Modulation depth")
    plt.title(f"Modulation depth vs p0 at (x={x_coord}, y={y_coord})")
    plt.tight_layout()

    save_path = output_dir / save_name
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    args = parse_args()

    input_file = Path(args.input_root) / args.filename / "input.in"
    if not input_file.exists():
        raise FileNotFoundError(f"Missing input file: {input_file}")

    nodes, num_crit = read_input_metadata(input_file)
    xi, yi = compute_indices(args.x, args.y, nodes, num_crit)

    output_dir = Path(args.output_root) / args.filename
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_output_pairs(output_dir)

    results = []
    for s_path, psi_path, p0_value in pairs:
        _, mod_depth, peak_time = analyze_pair(
            s_path, psi_path, p0_value, nodes, (xi, yi)
        )
        print(
            f"{s_path.name}: peak at t={peak_time:.3e}, "
            f"p0={p0_value if p0_value is not None else 'unknown'}, "
            f"M={mod_depth:.3e}"
        )
        results.append((p0_value if p0_value is not None else np.nan, mod_depth))

    plot_results(results, output_dir, args.save_fig, args.x, args.y)


if __name__ == "__main__":
    main()
