# -*- coding: utf-8 -*-
"""
Created on Wed Oct 9 09:45:14 2025

@author: Kelsey O'Donnell, Gordon Robb
"""
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from analytic_predictors import analytic_delay_time, pump_threshold
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    analytic_helpers = PROJECT_ROOT / "BEC_self_organisation"
    if analytic_helpers.exists():
        sys.path.append(str(analytic_helpers))
        from analytic_predictors import analytic_delay_time, pump_threshold
    else:
        raise

MAX_ANALYTIC_TIME = 6e10
TIME_EXTENSION_FACTOR = 2.5 

INPUT_FILE = None
OUTPUT_DIR = None
S_BASE = None
PSI_BASE = None
CURRENT_S_PATH = None
CURRENT_PSI_PATH = None

#Open new output data files
def openfiles():
    if CURRENT_S_PATH is None or CURRENT_PSI_PATH is None:
        raise RuntimeError("Output paths have not been configured.")

    CURRENT_S_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_PSI_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CURRENT_S_PATH.open("w"):
        pass

    with CURRENT_PSI_PATH.open("w"):
        pass

#Read input data from file
def readinput():
    data0 = np.genfromtxt(str(INPUT_FILE),skip_footer=1,comments='!')         #load input data file

    if data0.size not in (11, 12):
        raise ValueError(
            "Expected 11 or 12 numeric entries in input file (excluding final flag). "
            f"Found {data0.size} values."
        )

    nodes_per_dim=data0[0].astype(int)     
    maxt=data0[1]      
    ht=data0[2]
    width_psi=data0[3]
    p0_in=data0[4]
    Delta=data0[5]
    omega_r=data0[6]
    b0=data0[7]
    num_crit=data0[8]
    R=data0[9]
    plotnum=data0[10].astype(int)
    seed_amp = float(data0[11]) if data0.size >= 12 else 1.0e-6
    
    return nodes_per_dim,maxt,ht,width_psi,p0_in,Delta,omega_r,b0,num_crit,R,plotnum,seed_amp

#Initialise variables
def initvars():
    shift=np.pi/2.0
    L_dom=2.0*np.pi*num_crit
    hx=L_dom/np.float32(nodes_per_dim) 
    tperplot=maxt/np.float32(plotnum-1)
    x=np.linspace(0,L_dom-hx,nodes_per_dim)-L_dom/2.0
    xmat=np.tile(x,[nodes_per_dim,1])
    ymat=np.transpose(xmat)
    y0=np.complex64(np.exp(-(xmat**2+ymat**2)/(2.0*width_psi**2)))
    y0=y0*(1.0+seed_amplitude*(np.cos(xmat)+np.cos(-1.0/2.0*xmat+np.sqrt(3)/2.0*ymat)+np.cos(-1.0/2.0*xmat-np.sqrt(3)/2.0*ymat)));
    norm=hx*hx*np.sum(np.abs(y0)**2)
    y0=y0/np.sqrt(norm)*L_dom
    noise=np.random.random_sample(nodes_per_dim)*0.0
    kx=np.fft.fftfreq(nodes_per_dim, d=hx)*2.0*np.pi
    kxmat=np.tile(kx,[nodes_per_dim,1])
    kymat=np.transpose(kxmat)

    return shift,L_dom,hx,tperplot,x,y0,noise,kxmat,kymat

#Write data to output files
def output(t,y):
    psi=y
    F=np.sqrt(p0)*np.exp(-1j*b0/(2.0*Delta)*np.abs(psi)**2)*(1.0+noise)
    B=calc_B(F,shift)
    s=p0+np.abs(B)**2
    error=hx*hx*np.sum((np.abs(psi))**2)-L_dom**2
    mod=np.max(s)-np.min(s)
    
    f_s = open(CURRENT_S_PATH,"a+")
    data=np.insert((np.reshape(s,[1,nodes_per_dim*nodes_per_dim])),0,t)
    np.savetxt(f_s,data.reshape((1,nodes_per_dim*nodes_per_dim+1)), fmt='%1.3E',delimiter=' ')
    f_s.close()
    
    f_psi = open(CURRENT_PSI_PATH,"a+")
    data=np.insert((np.reshape(np.abs(psi)**2,[1,nodes_per_dim*nodes_per_dim])),0,t)
    np.savetxt(f_psi,data.reshape((1,nodes_per_dim*nodes_per_dim+1)), fmt='%1.3E',delimiter=' ')
    f_psi.close()
    
    progress=np.int32(t/maxt*100)
    print('Completed '+str(progress)+' % :  mod = '+str(mod)+',  Error ='+str(error))


    return t,mod,error

#Integrate kinetic energy part of Schrodinger equation
def propagate_bec(y,tstep):
    psi=y
    psi_k=np.fft.fft(psi)
    psi_k=psi_k*np.exp(-1j*omega_r*(kxmat**2+kymat**2)*tstep)
    psi=np.fft.ifft(psi_k)
    
    return psi

#Propagate optical field in free space to calculate backward field (B)
def calc_B(F,shift):
    Fk=np.fft.fft(F)
    Bk=np.sqrt(R)*Fk*np.exp(-1j*(kxmat**2+kymat**2)*shift)
    B=np.fft.ifft(Bk)
    
    return B

#2nd order Runge-Kutta algorithm                      
def rk2(t,y):
    yk1=ht*dy(t,y)
    tt=t+0.5*ht
    yt=y+0.5*yk1
    yk2=ht*dy(tt,yt)
    newt=t+ht
    newy=y+yk2
    
    return newt,newy

#RHS of ODEs for integration of potential energy part of Schrodinger equation
def dy(t,y):
    psi=y  
    F=np.sqrt(p0)*np.exp(-1j*b0/(2.0*Delta)*(np.abs(psi))**2)*(1.0+noise)
    B=calc_B(F,shift)
    return -1j*Delta/4.0*(p0+np.abs(B)**2)*psi

##########
def parse_args():
    parser = argparse.ArgumentParser(description="Run a 2D SMF simulation.")
    parser.add_argument(
        "-f",
        "--filename",
        metavar="simulation",
        required=True,
        help="Simulation name that identifies input and output folders.",
    )
    parser.add_argument(
        "-n",
        "--num-pump-frames",
        metavar="num_pump_frames",
        type=int,
        default=1,
        help="Number of pump saturation parameters to evaluate between start and end (default: 1).",
    )
    parser.add_argument(
        "-s",
        "--start-pump-param",
        metavar="start_pump_param",
        type=float,
        help="Starting pump saturation parameter p0.",
    )
    parser.add_argument(
        "-e",
        "--end-pump-param",
        metavar="end_pump_param",
        type=float,
        help="Ending pump saturation parameter p0.",
    )
    parser.add_argument(
        "-i",
        "--index",
        metavar="index",
        type=int,
        help="Optional frame index when distributing work across jobs.",
    )
    parser.add_argument(
        "--extend-time-using-t0",
        action="store_true",
        help="Extend simulation time using analytic delay-time estimates (similar to 1D script).",
    )
    parser.add_argument(
        "--input-root",
        default="inputs",
        help="Root directory containing simulation inputs (default: inputs).",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root directory where outputs will be written (default: outputs).",
    )
    return parser.parse_args()


def configure_paths(simulation_name, input_root, output_root):
    input_dir = Path(input_root) / simulation_name
    output_dir = Path(output_root) / simulation_name

    input_file = input_dir / "input.in"
    if not input_file.exists():
        raise FileNotFoundError(
            f"Could not find input file at {input_file}. "
            "Ensure the simulation folder and input.in exist."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    return input_file, output_dir


def build_output_paths(name_modifier):
    base_s = str(S_BASE)
    base_psi = str(PSI_BASE)
    suffix = name_modifier if name_modifier else ""
    s_path = Path(base_s + suffix + ".out")
    psi_path = Path(base_psi + suffix + ".out")
    return s_path, psi_path


def main():
    args = parse_args()

    if args.num_pump_frames < 1:
        raise ValueError("num_pump_frames must be at least 1.")

    if (args.start_pump_param is None) ^ (args.end_pump_param is None):
        raise ValueError("Specify both start and end pump parameters together.")

    global INPUT_FILE, OUTPUT_DIR, S_BASE, PSI_BASE, CURRENT_S_PATH, CURRENT_PSI_PATH
    INPUT_FILE, OUTPUT_DIR = configure_paths(
        args.filename, args.input_root, args.output_root
    )
    S_BASE = OUTPUT_DIR / "s"
    PSI_BASE = OUTPUT_DIR / "psi"

    global nodes_per_dim, maxt, ht, width_psi, p0, Delta, omega_r, b0, num_crit, R, plotnum, seed_amplitude
    (
        nodes_per_dim,
        maxt,
        ht,
        width_psi,
        p0_input,
        Delta,
        omega_r,
        b0,
        num_crit,
        R,
        plotnum,
        seed_amplitude,
    ) = readinput()

    base_maxt = float(maxt)
    base_p0 = float(p0_input)
    base_plotnum = int(plotnum)
    base_seed = float(seed_amplitude)

    start_param = args.start_pump_param
    end_param = args.end_pump_param
    num_frames = int(args.num_pump_frames)
    index_value = args.index

    if num_frames > 1 and (start_param is None or end_param is None):
        raise Exception(
            "If you specify more than one frame you must specify a start and end pump parameter"
        )

    if start_param is not None and end_param is not None:
        start_param = float(start_param)
        end_param = float(end_param)

    if start_param is None:
        pump_params = np.array([base_p0], dtype=float)
    else:
        if index_value is None:
            pump_params = np.linspace(start_param, end_param, num_frames, dtype=float)
        else:
            idx = int(index_value)
            num_intervals = num_frames
            t_fraction = idx / num_intervals if num_intervals > 0 else 0.0
            p_val = start_param + (end_param - start_param) * t_fraction
            pump_params = np.array([p_val], dtype=float)

    pump_params = np.atleast_1d(pump_params).astype(float)

    counter = int(index_value) if index_value is not None else 0

    extend_time_using_t0 = bool(args.extend_time_using_t0)
    pump_threshold_val = (
        pump_threshold(omega_r, b0, R) if extend_time_using_t0 else None
    )

    global shift, L_dom, hx, tperplot, x, y0, noise, kxmat, kymat

    for pump_param in pump_params:
        p0 = float(pump_param)

        name_modifier = ""
        if num_frames > 1 or index_value is not None:
            name_modifier = str(counter) + "_" + str(p0)

        CURRENT_S_PATH, CURRENT_PSI_PATH = build_output_paths(name_modifier)

        openfiles()

        maxt = base_maxt
        plotnum = base_plotnum

        shift, L_dom, hx, tperplot, x, y0, noise, kxmat, kymat = initvars()

        if extend_time_using_t0:
            target_maxt = base_maxt
            if (
                pump_threshold_val is not None
                and np.isfinite(pump_threshold_val)
                and p0 > pump_threshold_val
            ):
                analytic_t0 = analytic_delay_time(
                    p0, pump_threshold_val, omega_r, base_seed
                )
                if np.isfinite(analytic_t0):
                    adjusted_time = min(
                        analytic_t0 * TIME_EXTENSION_FACTOR, MAX_ANALYTIC_TIME
                    )
                    target_maxt = adjusted_time 

            if target_maxt - base_maxt > 1e-12:
                print(
                    "Extending simulation time for p0="
                    + f"{p0:.3e}"
                    + " from "
                    + f"{base_maxt:.3e}"
                    + " to "
                    + f"{target_maxt:.3e}"
                )

            maxt = target_maxt
            if plotnum > 1:
                tperplot = maxt / np.float32(plotnum - 1)
            else:
                tperplot = maxt

        y = np.array(y0, copy=True)
        t = 0.0
        nextt = tperplot
        ind = 0
        output(t, y)
        while t < maxt:
            y = propagate_bec(y, 0.5 * ht)
            t, y = rk2(t, y)
            y = propagate_bec(y, 0.5 * ht)
            if t >= nextt:
                output(t, y)
                ind = ind + 1
                nextt = nextt + tperplot

        counter += 1
        print("Finished " + str(counter) + "/" + str(num_frames) + " frames")
    print("Finished all frames!")


if __name__ == "__main__":
    main()
