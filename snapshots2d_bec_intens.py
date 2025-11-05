#Plots output from code PATT1D_Q_SFM_FFT_S
# Plots intensities and phases of Optical field and BEC wavefunction.
#Generates sequence of .png files 

# -*- coding: utf-8 -*-
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

#fname = raw_input("Enter filename: ")
plt.rcParams['ps.usedistiller']='xpdf'                            # improves quality of .eps figures for use with LaTeX

def parse_args():
    parser = argparse.ArgumentParser(description="Generate 2D intensity snapshots.")
    parser.add_argument(
        "-f",
        "--filename",
        metavar="simulation",
        required=True,
        help="Simulation name that identifies input/output folders.",
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
        "--frames-dir",
        default="frames",
        help="Name of the subdirectory for rendered frames (default: frames).",
    )
    return parser.parse_args()

def configure_paths(simulation, input_root, output_root, frames_dir):
    input_dir = Path(input_root) / simulation
    output_dir = Path(output_root) / simulation

    input_file = input_dir / "input.in"
    if not input_file.exists():
        raise FileNotFoundError(f"Could not find input file at {input_file}.")

    s_path = output_dir / "s.out"
    psi_path = output_dir / "psi.out"
    if not s_path.exists() or not psi_path.exists():
        raise FileNotFoundError(
            f"Expected simulation outputs at {s_path} and {psi_path}; run the simulation first."
        )

    frame_dir = output_dir / frames_dir
    frame_dir.mkdir(parents=True, exist_ok=True)
    return input_file, s_path, psi_path, frame_dir

def load_outputs(s_path, psi_path):
    print ('Loading '+str(s_path))
    data1 = np.loadtxt(str(s_path))                                       #load dataset in the form t, intensity
    print ('Loading '+str(psi_path))
    data2 = np.loadtxt(str(psi_path))                                       #load dataset in the form t, |Psi|^2
    return data1, data2


#Read input data from file
def readinput(fname0):
    data0 = np.genfromtxt(str(fname0),skip_footer=1,comments='!')         #load input data file

    nodes_per_dim=data0[0].astype(int)     
    maxt=data0[1]      
    ht=data0[2]
    width_psi=data0[3]
    p0=data0[4]
    Delta=data0[5]
    gambar=data0[6]
    b0=data0[7]
    num_crit=data0[8]
    R=data0[9]
    plotnum=data0[10].astype(int)
    
    return nodes_per_dim,maxt,ht,width_psi,p0,Delta,gambar,b0,num_crit,R,plotnum


def main():
    args = parse_args()
    input_file, s_path, psi_path, frame_dir = configure_paths(
        args.filename, args.input_root, args.output_root, args.frames_dir
    )

    data1, data2 = load_outputs(s_path, psi_path)

    nodes_per_dim,maxt,ht,width_psi,p0,Delta,gambar,b0,num_crit,R,plotnum=readinput(input_file)
    t = data1[:,0] 
    plotnum=len(t)

#plt.ion()
    plt.ioff()

    pi=4.0*np.arctan(1.0)
    xco=np.linspace(-pi*num_crit,pi*num_crit,nodes_per_dim)

    fig=plt.figure()
    step=1
    count=0
    for j in np.arange (0,plotnum-1,step):
        count=count+1
        print ('Generating frame '+str(count))
        
        s=np.reshape(data1[j,1:],[nodes_per_dim,nodes_per_dim]) 
        prob=np.reshape(data2[j,1:],[nodes_per_dim,nodes_per_dim]) 


        fig.suptitle(r'$\Gamma_2$ t='+str('%.2e'%t[j]), fontsize=14)

   
        ax1=plt.subplot(121,aspect='equal')
        f1=ax1.imshow(prob, extent=[-pi*num_crit,pi*num_crit,-pi*num_crit,pi*num_crit], origin="lower", vmin=prob.min(),vmax=prob.max(), aspect="auto", cmap='bone')
        cb1=fig.colorbar(f1,orientation='horizontal')
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_title('BEC density $|\\Psi|^2$',fontsize=14)
        ax1.set_axis_off()

        ax2=plt.subplot(122,aspect='equal')
        f2=ax2.imshow(s, extent=[-pi*num_crit,pi*num_crit,-pi*num_crit,pi*num_crit], origin="lower", vmin=s.min(),vmax=s.max(), aspect="auto", cmap='bone')
        cb1=fig.colorbar(f2,orientation='horizontal')
        ax2.set_title('Intensity (s)',fontsize=14)
        ax2.set_axis_off()

        plt.tight_layout()
        plt.draw()
#    plt.show()
#    time.sleep(0.02)
        filename=frame_dir/ (str('%03d' %count) + '.png')
        fig.savefig(str(filename), dpi=200)
        plt.clf()
   
#plt.ioff()

if __name__ == "__main__":
    main()
