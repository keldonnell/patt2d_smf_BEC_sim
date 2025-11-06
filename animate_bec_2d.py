#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create an animation of the 2D BEC simulation showing density and intensity side by side.
"""
import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate BEC density and intensity for a 2D simulation."
    )
    parser.add_argument(
        "-f",
        "--filename",
        metavar="simulation",
        required=True,
        help="Simulation name identifying the input/output folders.",
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
        "-o",
        "--output-file",
        default="animation.mp4",
        help="Path to write the animation (default: animation.mp4).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for the animation (default: 15).",
    )
    parser.add_argument(
        "--writer",
        choices=["ffmpeg", "pillow"],
        default="ffmpeg",
        help="Animation writer to use (default: ffmpeg).",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        help="Optional pump-frame index to select when multiple output files exist.",
    )
    return parser.parse_args()


def pick_output_files(output_dir: Path, index: Optional[int]):
    base_s = output_dir / "s.out"
    base_psi = output_dir / "psi.out"

    def find_indexed(idx: int):
        s_candidates = sorted(output_dir.glob(f"s{idx}_*.out"))
        psi_candidates = sorted(output_dir.glob(f"psi{idx}_*.out"))
        if not s_candidates or not psi_candidates:
            return None, None
        if len(s_candidates) > 1 or len(psi_candidates) > 1:
            raise RuntimeError(
                f"Found multiple files for index {idx} in {output_dir}; "
                "please clean up duplicates."
            )
        return s_candidates[0], psi_candidates[0]

    if index is not None:
        s_file, psi_file = find_indexed(int(index))
        if s_file is None or psi_file is None:
            raise FileNotFoundError(
                f"No output files found for index {index} in {output_dir}."
            )
        return s_file, psi_file

    if base_s.exists() and base_psi.exists():
        return base_s, base_psi

    s_map = {}
    for s_file in output_dir.glob("s*_*.out"):
        match = re.match(r"s(\d+)_", s_file.stem)
        if match:
            s_map[int(match.group(1))] = s_file

    psi_map = {}
    for psi_file in output_dir.glob("psi*_*.out"):
        match = re.match(r"psi(\d+)_", psi_file.stem)
        if match:
            psi_map[int(match.group(1))] = psi_file

    common_indices = sorted(set(s_map) & set(psi_map))
    if not common_indices:
        raise FileNotFoundError(
            f"Could not locate matching s/psi output files in {output_dir}."
        )
    if len(common_indices) > 1:
        raise RuntimeError(
            "Multiple indexed output files found. "
            "Please re-run with --index to choose a specific frame."
        )

    idx = common_indices[0]
    return s_map[idx], psi_map[idx]


def configure_paths(simulation, input_root, output_root, index: Optional[int]):
    input_dir = Path(input_root) / simulation
    output_dir = Path(output_root) / simulation

    input_file = input_dir / "input.in"
    if not input_file.exists():
        raise FileNotFoundError(f"Could not find input file at {input_file}.")

    s_path, psi_path = pick_output_files(output_dir, index)

    return input_file, s_path, psi_path, output_dir


def read_input_metadata(input_file):
    data0 = np.genfromtxt(str(input_file), skip_footer=1, comments="!")
    nodes_per_dim = int(data0[0])
    num_crit = data0[8]
    return nodes_per_dim, num_crit


def load_output_data(s_path, psi_path):
    s_data = np.loadtxt(str(s_path))
    psi_data = np.loadtxt(str(psi_path))
    if s_data.shape != psi_data.shape:
        raise ValueError("s.out and psi.out have incompatible shapes.")
    return s_data, psi_data


def prepare_axes(nodes_per_dim, num_crit, s_data, psi_data):
    times = s_data[:, 0]
    s_values = s_data[:, 1:]
    psi_values = psi_data[:, 1:]

    s_min, s_max = np.min(s_values), np.max(s_values)
    psi_min, psi_max = np.min(psi_values), np.max(psi_values)

    pi = 4.0 * np.arctan(1.0)
    extent = [-pi * num_crit, pi * num_crit, -pi * num_crit, pi * num_crit]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    im_psi = axes[0].imshow(
        psi_values[0].reshape(nodes_per_dim, nodes_per_dim),
        extent=extent,
        origin="lower",
        vmin=psi_min,
        vmax=psi_max,
        cmap="bone",
        animated=True,
    )
    axes[0].set_title(r"BEC density $|\Psi|^2$")
    fig.colorbar(im_psi, ax=axes[0], orientation="vertical")

    im_s = axes[1].imshow(
        s_values[0].reshape(nodes_per_dim, nodes_per_dim),
        extent=extent,
        origin="lower",
        vmin=s_min,
        vmax=s_max,
        cmap="bone",
        animated=True,
    )
    axes[1].set_title("Intensity (s)")
    fig.colorbar(im_s, ax=axes[1], orientation="vertical")

    time_text = axes[0].text(
        0.02,
        0.95,
        "",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color="white",
        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
    )

    fig.tight_layout()

    return fig, axes, im_psi, im_s, times, psi_values, s_values, time_text


def animate(
    simulation,
    input_root,
    output_root,
    index: Optional[int],
    output_file,
    fps,
    writer_choice,
):
    input_file, s_path, psi_path, output_dir = configure_paths(
        simulation, input_root, output_root, index
    )
    print(f"Using output files:\n  {s_path}\n  {psi_path}")
    nodes_per_dim, num_crit = read_input_metadata(input_file)
    s_data, psi_data = load_output_data(s_path, psi_path)

    fig, axes, im_psi, im_s, times, psi_values, s_values, time_text = prepare_axes(
        nodes_per_dim, num_crit, s_data, psi_data
    )

    title = fig.suptitle("", fontsize=14)

    def update(frame_index):
        psi_frame = psi_values[frame_index].reshape(nodes_per_dim, nodes_per_dim)
        s_frame = s_values[frame_index].reshape(nodes_per_dim, nodes_per_dim)

        im_psi.set_array(psi_frame)
        im_s.set_array(s_frame)

        current_time = times[frame_index]
        title.set_text(r"$\Gamma_2 t = {:.2e}$".format(current_time))
        time_text.set_text(f"Time: {current_time:.2e}")
        return im_psi, im_s, title, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=1000.0 / fps,
        blit=True,
    )

    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = output_dir / output_path

    if writer_choice == "ffmpeg":
        writer = animation.FFMpegWriter(fps=fps)
    else:
        writer = animation.PillowWriter(fps=fps)

    print(f"Saving animation to {output_path}")
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    print("Animation complete.")


def main():
    args = parse_args()
    animate(
        args.filename,
        args.input_root,
        args.output_root,
        args.index,
        args.output_file,
        args.fps,
        args.writer,
    )


if __name__ == "__main__":
    main()
