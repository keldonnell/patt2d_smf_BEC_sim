# -*- coding: utf-8 -*-
"""
Created on Thu Oct 8 09:45:14 2020

@author: Gordon Robb
"""
import argparse
import os

import numpy as np

from analytic_predictors import analytic_delay_time, pump_threshold

MAX_ANALYTIC_TIME = 1e11
TIME_EXTENSION_FACTOR = 3.0 

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "-f",
    "--filename",
    metavar="filename",
    required=True,
    help="The name of the file to save to",
)

parser.add_argument(
    "-n",
    "--num_pump_frames",
    metavar="num_pump_frames",
    required=True,
    help="The number of different pump saturation param generated frames",
)

parser.add_argument(
    "-s",
    "--start_pump_param",
    metavar="start_pump_param",
    required=False,
    help="The starting pump saturation param",
)

parser.add_argument(
    "-e",
    "--end_pump_param",
    metavar="end_pump_param",
    required=False,
    help="The ending pump saturation param",
)

parser.add_argument(
    "-i",
    "--index",
    metavar="index",
    required=False,
    help="The index of the p_0 value. This is used for SLURM",
)

parser.add_argument(
    "--density-centers",
    metavar="p0",
    nargs="+",
    type=float,
    required=False,
    help="Optional list (max 2) of p0 values where additional Gaussian density should be applied.",
)

parser.add_argument(
    "--density-width",
    metavar="width",
    nargs="+",
    type=float,
    required=False,
    help="Standard deviation(s) of the Gaussian density (in p0 units). Provide one value or one per centre; defaults to (end-start)/20.",
)

parser.add_argument(
    "--density-strength",
    metavar="strength",
    type=float,
    default=5.0,
    help="Relative strength of the Gaussian density enhancement (ignored if no centers supplied).",
)

parser.add_argument(
    "--extend-time-using-t0",
    action="store_true",
    help="Extend simulation time to 1.5 * t0 (capped at 1e11) for p0 values above threshold.",
)

args = parser.parse_args()

density_centers_config = list(args.density_centers) if args.density_centers else []
density_width_config = list(args.density_width) if args.density_width else []
density_strength_config = args.density_strength if args.density_strength is not None else 0.0
extend_time_using_t0 = bool(args.extend_time_using_t0)

if len(density_centers_config) > 2:
    raise Exception("You may specify at most two density centres.")

if density_centers_config and (
    args.start_pump_param is None or args.end_pump_param is None
):
    raise Exception(
        "Gaussian density enhancement requires both start and end pump parameters."
    )

if int(args.num_pump_frames) > 1 and (
    args.start_pump_param == None or args.end_pump_param == None
):
    raise Exception(
        "If you specify more than one frame you must specify a start and end pump parameter"
    )

output_dir = "patt1d_outputs/" + args.filename + "/"
input_dir = "patt1d_inputs/" + args.filename + "/"
s_dir = output_dir + "s"
psi_dir = output_dir + "psi"
seed_dir = input_dir + "seed.in"


import os
from pathlib import Path

if not(os.path.exists(output_dir)) and os.path.exists(input_dir):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)  # instead of os.mkdir(...)
        print("THIS SHOULD ONLY BE REACHED ONCE")
elif os.path.exists(output_dir) and args.index == None:
	raise Exception("That filename already exits")

	
	


""" # Open new output data files
def openfiles():

    f_s = open(s_dir, "w")
    f_s.close()

    f_psi = open(psi_dir, "w")
    f_psi.close() """


# Read input data from file
def readinput():
    data0 = np.genfromtxt(seed_dir, skip_footer=1, comments="!")  # load input data file

    nodes = data0[0].astype(int)
    maxt = data0[1]
    ht = data0[2]
    width_psi = data0[3]
    p0 = data0[4]
    Delta = data0[5]
    omega_r = data0[6]
    b0 = data0[7]
    num_crit = data0[8]
    R = data0[9]
    gbar = data0[10]
    v0 = data0[11]
    plotnum = data0[12].astype(int)
    seed = data0[13]
    noise1 = data0[14]
    noise2 = data0[15]
    noise3 = data0[16]

    return (
        nodes,
        maxt,
        ht,
        width_psi,
        p0,
        Delta,
        omega_r,
        b0,
        num_crit,
        R,
        gbar,
        v0,
        plotnum,
        seed,
        noise1,
        noise2,
        noise3
    )


# Initialise variables
def initvars():

    shift = np.pi / 2.0
    L_dom = 2.0 * np.pi * num_crit
    hx = L_dom / np.float32(nodes)
    tperplot = maxt / np.float32(plotnum - 1)
    x = np.linspace(0, L_dom - hx, nodes) - L_dom / 2.0
    y0 = np.complex64(np.exp(-(x**2) / (2.0 * width_psi**2)))
    y0 = y0 * np.exp(1j * v0 * x)

    noise1_vals = (
        np.random.uniform(-1, 1, nodes) * noise1 
    )  
    noise2_vals = (
        np.random.uniform(-1, 1, nodes) * noise2 
    )  
    noise3_vals = (
        np.random.uniform(-1, 1, nodes) * noise3 
    )  
    
    seed_vals = seed * np.cos(x)

    y0 = y0 * (np.ones(nodes) + seed_vals + noise1_vals + noise2_vals)
    norm = hx * np.sum(np.abs(y0) ** 2)
    y0 = y0 / np.sqrt(norm) * np.sqrt(L_dom)


    kx = np.fft.fftfreq(nodes, d=hx) * 2.0 * np.pi

    return shift, L_dom, hx, tperplot, x, y0, kx, noise3_vals


# Write data to output files
def output(t, y, p0, counter):
    name_modifier = ""
    if int(args.num_pump_frames) > 1 or args.index != None:
        name_modifier = str(counter) + "_" + str(p0)

    psi = y

    F = (
        np.sqrt(p0)
        * np.exp(-1j * b0 / (2.0 * Delta) * np.abs(psi) ** 2)
    )

    B = calc_B(F, shift)
    s = p0 + np.abs(B) ** 2
    error = hx * np.sum((np.abs(psi)) ** 2) - L_dom
    mod = np.max(s) - np.min(s)

    f_s = open(s_dir + name_modifier + ".out", "a+")
    data = np.concatenate(([t], s))
    np.savetxt(f_s, data.reshape((1, nodes + 1)), fmt="%1.3E", delimiter=" ")
    f_s.close()

    f_psi = open(psi_dir + name_modifier + ".out", "a+")
    data = np.concatenate(([t], np.abs(psi) ** 2))
    np.savetxt(f_psi, data.reshape((1, nodes + 1)), fmt="%1.3E", delimiter=" ")
    f_psi.close()

    progress = np.int32(t / maxt * 100)
    print(
        "Completed "
        + str(progress)
        + " % :  mod = "
        + str(mod)
        + ",  Error ="
        + str(error)
    )

    return t, mod, error


# Integrate kinetic energy part of Schrodinger equation
def propagate_bec(y, tstep):
    psi = y
    psi_k = np.fft.fft(psi)
    psi_k = psi_k * np.exp(-1j * omega_r * kx**2 * tstep)
    psi = np.fft.ifft(psi_k)

    return psi


# Propagate optical field in free space to calculate backward field (B)
def calc_B(F, shift):
    Fk = np.fft.fft(F)
    Bk = np.sqrt(R) * Fk * np.exp(-1j * kx**2 * shift)
    B = np.fft.ifft(Bk)

    return B


def gaussian_weighted_spacing(num_points, start, end, centres, sigmas, strength):
    """
    Construct a set of sample points between start and end with enhanced density near centres.
    """
    if num_points <= 1:
        return np.array([start], dtype=float)

    start_val = float(start)
    end_val = float(end)

    centres = np.asarray(centres, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)

    if centres.shape != sigmas.shape:
        raise ValueError("Centres and sigmas must have the same shape.")

    if np.any(sigmas <= 0):
        raise ValueError("Gaussian density widths must be positive.")

    # Work with monotonically increasing domain and flip back if necessary.
    flip = False
    if end_val < start_val:
        start_val, end_val = end_val, start_val
        flip = True

    sample_count = max(2000, num_points * 40)
    x_grid = np.linspace(start_val, end_val, sample_count)
    weights = np.ones_like(x_grid)

    if strength > 0 and centres.size:
        for centre, sigma in zip(centres, sigmas):
            weights += strength * np.exp(-0.5 * ((x_grid - centre) / sigma) ** 2)

    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]

    target_probs = np.linspace(0.0, 1.0, num_points)
    values = np.interp(target_probs, cumulative, x_grid)

    if flip:
        values = values[::-1]

    values[0] = float(start)
    values[-1] = float(end)

    return values


# 2nd order Runge-Kutta algorithm
def rk2(t, y, p0):
    yk1 = ht * dy(t, y, p0)
    tt = t + 0.5 * ht
    yt = y + 0.5 * yk1
    yk2 = ht * dy(tt, yt, p0)
    newt = t + ht
    newy = y + yk2

    return newt, newy


# RHS of ODEs for integration of potential energy part of Schrodinger equation
def dy(t, y, p0):
    psi = y
    F = (
        np.sqrt(p0)
        * np.exp(-1j * b0 / (2.0 * Delta) * (np.abs(psi)) ** 2)
    )
    B = calc_B(F, shift)
    return -1j * Delta / 4.0 * (p0 + np.abs(B) ** 2) * psi


##########

""" openfiles() """
(
    nodes,
    maxt,
    ht,
    width_psi,
    p0,
    Delta,
    omega_r,
    b0,
    num_crit,
    R,
    gbar,
    v0,
    plotnum,
    seed,
    noise1,
    noise2,
    noise3
) = readinput()
shift, L_dom, hx, tperplot, x, y0, kx, noise3_vals = initvars()
base_maxt = float(maxt)
base_tperplot = float(tperplot)
gamma_bar = omega_r  # Reuse the growth-rate parameter used in analytic estimates.
pump_threshold_val = pump_threshold(gamma_bar, b0, R) if extend_time_using_t0 else None

num_pump_frames = int(args.num_pump_frames)
start_param = float(args.start_pump_param) if args.start_pump_param is not None else None
end_param = float(args.end_pump_param) if args.end_pump_param is not None else None

density_centres = list(density_centers_config)
density_strength = float(density_strength_config) if density_strength_config is not None else 0.0
density_widths = list(density_width_config)

gaussian_sigmas = []
if start_param is not None and end_param is not None:
    range_min = min(start_param, end_param)
    range_max = max(start_param, end_param)

    if density_centres:
        clipped_centres = []
        for centre in density_centres:
            if centre < range_min or centre > range_max:
                print(
                    "Warning: density centre "
                    + str(centre)
                    + " lies outside the requested range; clipping."
                )
                centre = min(max(centre, range_min), range_max)
            clipped_centres.append(centre)
        density_centres = sorted(clipped_centres)

        if density_widths:
            if len(density_widths) == 1:
                sigma_val = abs(density_widths[0])
                gaussian_sigmas = [sigma_val] * len(density_centres)
            elif len(density_widths) == len(density_centres):
                gaussian_sigmas = [abs(val) for val in density_widths]
            else:
                raise Exception(
                    "Provide either one density width or one per density centre."
                )
        else:
            span = abs(end_param - start_param)
            if span <= 0:
                raise Exception(
                    "Cannot determine a default density width when start and end are equal."
                )
            default_sigma = span / 20.0
            gaussian_sigmas = [default_sigma] * len(density_centres)

density_active = (
    bool(density_centres)
    and bool(gaussian_sigmas)
    and start_param is not None
    and end_param is not None
    and density_strength > 0
)

if density_active:
    if any(sigma <= 0 for sigma in gaussian_sigmas):
        raise Exception("Gaussian density widths must be positive and non-zero.")

if num_pump_frames > 1 and args.index == None:
    if density_active:
        pump_params = gaussian_weighted_spacing(
            num_pump_frames,
            start_param,
            end_param,
            density_centres,
            gaussian_sigmas,
            density_strength,
        )
    else:
        pump_params = np.linspace(start_param, end_param, num_pump_frames)
elif (
    start_param is not None
    and end_param is not None
    and args.index != None
):
    idx = int(args.index)
    if density_active:
        params = gaussian_weighted_spacing(
            num_pump_frames,
            start_param,
            end_param,
            density_centres,
            gaussian_sigmas,
            density_strength,
        )
        if idx < 0 or idx >= len(params):
            raise IndexError(
                f"Index {idx} is outside the valid range [0, {len(params) - 1}]."
            )
        pump_params = [params[idx]]
    else:
        start = start_param
        end = end_param
        num_intervals = num_pump_frames

        if num_intervals > 0:
            t = idx / num_intervals
        else:
            t = 0.0

        p_val = start + (end - start) * t
        pump_params = [p_val]
else:
    pump_params = [p0]

pump_params = np.atleast_1d(pump_params).astype(float)


if args.index == None:
    counter = 0
else:
    counter = int(args.index)


for pump_param in pump_params:
    p0 = float(pump_param)
    maxt = base_maxt
    tperplot = base_tperplot

    if extend_time_using_t0:
        target_maxt = base_maxt
        if pump_threshold_val is not None and np.isfinite(pump_threshold_val) and p0 > pump_threshold_val:
            analytic_t0 = analytic_delay_time(p0, pump_threshold_val, gamma_bar, seed)
            if np.isfinite(analytic_t0):
                adjusted_time = min(analytic_t0 * TIME_EXTENSION_FACTOR, MAX_ANALYTIC_TIME)
                target_maxt = max(base_maxt, adjusted_time)
        maxt = target_maxt
        if plotnum > 1:
            tperplot = maxt / np.float32(plotnum - 1)
        else:
            tperplot = maxt
        if maxt - base_maxt > 1e-12:
            print(
                "Extending simulation time for p0="
                + f"{p0:.3e}"
                + " from " 
                + f"{base_maxt}"
                + " to "
                + f"{maxt:.3e}"
            )

    y = y0
    t = 0.0
    nextt = tperplot
    ind = 0
    output(t, y, p0, counter)

    while t < maxt:
        
        noise2_vals = (
            np.random.uniform(-1, 1, nodes) * noise2 
        )  
        y = y * (np.ones(nodes) + noise2_vals)

        y = propagate_bec(y, 0.5 * ht)
        t, y = rk2(t, y, p0)
        y = propagate_bec(y, 0.5 * ht)
        if t >= nextt:
            output(t, y, p0, counter)
            ind = ind + 1
            nextt = nextt + tperplot
    counter += 1

    print("Finished " + str(counter) + "/" + str(args.num_pump_frames) + " frames")
print("Finished all frames!")
