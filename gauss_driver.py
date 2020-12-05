import sys
from driver import GaussDriver, ExitSignal, TimeOutSignal
import os
from mpi4py import MPI

# Set up MPI environment
comm = MPI.COMM_WORLD
nproc = comm.Get_size()   # Size of communicator
iproc = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs
shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
shm_rank = shm_comm.rank # rank index in sub-comm


def get_atomlist(xyz_name):
    atom_list = []
    with open(xyz_name, 'r') as xyz:
        lines = xyz.readlines()
        for idx, l in enumerate(lines):
            if idx > 1:
                atom_list.append(l.split()[0])
    print(atom_list)
    return atom_list


xyz_name = sys.argv[1]
# ("xyz name: ", xyz_name)
atom_list = get_atomlist(xyz_name)
# print("Atom list: ", atom_list)
port = int(os.environ.get("port", 31415))
driver = GaussDriver(port, "127.0.0.1", "template.gjf", atom_list)
while True:
    try:
        driver.parse()
    except ExitSignal as e:
        driver = GaussDriver(port, "127.0.0.1", "template.gjf", atom_list)
    except TimeOutSignal as e:
        if iproc == 0:
            print("Time out. Check whether the server is closed.")
        exit()
