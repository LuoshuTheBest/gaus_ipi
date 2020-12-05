import sys
import os
import itertools
import numpy as np
from mpi4py import MPI

# la
# Set up MPI environment
comm = MPI.COMM_WORLD
nproc = comm.Get_size()  # Size of communicator
iproc = comm.Get_rank()  # Ranks in communicator
inode = MPI.Get_processor_name()  # Node where this MPI process runs
shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)  # Create sub-comm for each node
shm_rank = shm_comm.rank  # rank index in sub-comm


def get_SideLength(xyz_name):
    with open(xyz_name, 'r') as xyz:
        lines = xyz.readlines()
        coord_list = []
        for idx, l in enumerate(lines):
            if idx > 1:
                coord_i = []
                for i in l.split()[1:]:
                    coord_i.append(float(i))
                coord_list.append(coord_i)
        idx_list = range(len(coord_list))
        atom_dist = []
        for idx0, co0 in enumerate(coord_list):
            idx1_list = idx_list[idx0:]
            for idx1 in idx1_list:
                co1 = coord_list[idx1]
                atom_dist.append(np.linalg.norm(np.asarray(co1) - np.asarray(co0)))
    return max(atom_dist)


def gen_input(verbose, output_opt, stride, len_side, total_steps, port, nbeads,
              seed, temperature, pressure, dyn_mode, therm_mode, baro_mode, tau, time_step):
    text = ""
    text += "<simulation verbosity='%s'>\n" % verbose
    # Set up output
    text += "\t<output prefix='simulation'>\n"
    output_opt = output_opt.split()
    if 'prop' in output_opt:
        text += "\t\t<properties stride='%s' filename='out'>  [step, time{%s}, conserved{%s}, temperature{%s}, kinetic_cv{%s}, potential{%s}, pressure_cv{%s}] </properties>\n" % (
        stride, time_unit, temp_unit, temp_unit, temp_unit, potential_unit, press_unit)
    if 'pos' in output_opt:
        text += "\t\t<trajectory filename='pos' stride='%s' format='xyz' cell_units='%s'> positions{%s} </trajectory>\n" % (
        stride, cell_units, cell_units)
    if 'force' in output_opt:
        text += "\t\t<trajectory filename='force' stride='%s' format='xyz' cell_units='%s'> forces{%s} </trajectory>\n\t</output>\n" % (
        stride, cell_units, force_unit)
    if 'vel' in output_opt:
        text += "\t\t<trajectory filename='vel' stride='%s' format='xyz' cell_units='%s'> velocities </trajectory>\n" % (
        stride, cell_units)
    if 'chk' in output_opt:
        text += "\t\t<checkpoint filename='chk' stride='%s' overwrite='True'/>\n\t</output>\n" % (int(stride) * 10)
    # Set up steps
    text += "\t<total_steps> %s </total_steps>\n" % total_steps
    # Set up prng
    text += "\t<prng>\n\t\t<seed> %s </seed>\n\t</prng>\n" % seed
    # Set up socket
    text += "\t<ffsocket mode='inet' name='driver' pbc='False'>\n\t\t<address>localhost</address>\n\t\t<port> %s </port>\n\t</ffsocket>\n" % port
    # Set up system
    text += "\t<system>\n"
    # Set up initialize
    text += "\t\t<initialize nbeads='%s'>\n" % nbeads
    text += "\t\t\t<file mode='xyz' units='%s'> %s </file>\n" % (cell_units, xyz_name)
    text += "\t\t\t<cell mode='abc' units='%s'> [ %.1f, %.1f, %.1f ] </cell>\n" % (
    cell_units, len_side, len_side, len_side)
    text += "\t\t\t<velocities mode='thermal' units='%s'> %s </velocities>\n\t\t</initialize>\n" % (
    temp_unit, temperature)
    # Set up forces
    text += "\t\t<forces>\n\t\t\t<force forcefield='driver'/>\n\t\t</forces>\n"
    # Set up ensemble
    text += "\t\t<ensemble>\n\t\t\t<temperature units='%s'> %s </temperature>\n" % (temp_unit, temperature)
    if dyn_mode == 'npt':
        text += "\t\t\t<pressure units='%s'> %s </pressure>\n" % (press_unit, pressure)
    text += "\t\t</ensemble>\n"
    # Set up motion
    text += "\t\t<motion mode='dynamics'>\n\t\t\t<dynamics mode='%s'>\n" % dyn_mode
    if dyn_mode == 'npt':
        text += "\t\t\t\t<barostat mode='%s'>\n\t\t\t\t\t<tau units='%s'> %s </tau>\n" % (baro_mode, time_unit, tau)
        text += "\t\t\t\t\t<thermostat mode='%s'>\n\t\t\t\t\t\t<tau units='%s'> %s </tau>\n\t\t\t\t\t</thermostat>\n\t\t\t\t</barostat>\n" % (
        therm_mode, time_unit, tau)
    if dyn_mode == 'nvt' or dyn_mode == 'npt':
        text += "\t\t\t\t<thermostat mode='%s'>\n\t\t\t\t\t<tau units='%s'> %s </tau>\n\t\t\t\t</thermostat>\n" % (
        therm_mode, time_unit, tau)
    text += "\t\t\t\t<timestep units='%s'> %s </timestep>\n" % (time_unit, time_step)
    text += "\t\t\t</dynamics>\n\t\t</motion>\n\t</system>\n</simulation>"
    with open('input.xml', 'w') as f:
        f.write(text)


xyz_name = sys.argv[1]
len_side = get_SideLength(xyz_name) * 20
# Set up MD units
time_unit = os.environ.get("time_unit", 'femtosecond')
temp_unit = os.environ.get("temp_unit", 'kelvin')
potential_unit = os.environ.get("potential_unit", 'electronvolt')
press_unit = os.environ.get("press_unit", 'bar')
cell_units = os.environ.get("cell_units", 'angstrom')
force_unit = os.environ.get("force_unit", 'piconewton')
# Set up MD parameters
verbose = int(os.environ.get("verbose", 5))
if verbose == 5:
    verbose = 'high'
else:
    verbose = 'medium'
output_opt = os.environ.get("output_opt", 'prop')
stride = os.environ.get("stride", '20')
nbeads = os.environ.get("nbeads", '1')
port = int(os.environ.get("port", 31415))
total_steps = os.environ.get("total_steps", 1000)
seed = os.environ.get("seed", 3348)
temperature = os.environ.get("temperature", 25)
pressure = os.environ.get("pressure", 10)
dyn_mode = os.environ.get("dyn_mode", 'nvt')
therm_mode = os.environ.get("therm_mode", 'pile_g')
baro_mode = os.environ.get("baro_mode", 'isotropic')
tau = os.environ.get("tau", 10)
time_step = os.environ.get("time_step", 0.25)

# Generate input
if iproc == 0:
    gen_input(verbose=verbose, output_opt=output_opt, stride=stride, len_side=len_side, total_steps=total_steps,
              port=port,
              nbeads=nbeads, seed=seed, temperature=temperature, pressure=pressure, dyn_mode=dyn_mode,
              therm_mode=therm_mode, baro_mode=baro_mode, tau=tau, time_step=time_step)
comm.Barrier()
