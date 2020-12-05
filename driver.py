"""
---------------------------------------------------------------------
|I-PI socket client.
|
|Version: 0.1
|Program Language: Python 3.6
|Developer: Xinyan Wang
|Homepage:https://github.com/WangXinyan940/i-pi-driver
|
|Receive coordinate and send force back to i-PI server using socket.
|Read http://ipi-code.org/assets/pdf/manual.pdf for details.
---------------------------------------------------------------------
"""
import os
import socket
import struct
import numpy as np
import ctypes
import sys
import psutil
import numpy as np
from pkg_resources import resource_filename
import subprocess
from mpi4py import MPI

#Set up MPI environment
comm = MPI.COMM_WORLD
nproc = comm.Get_size()   # Size of communicator
iproc = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs
shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
shm_rank = shm_comm.rank # rank index in sub-comm


# CONSTANTS
BOHR = 5.291772108e-11  # Bohr -> m
ANGSTROM = 1e-10  # angstrom -> m
AMU = 1.660539040e-27  # amu -> kg
FEMTO = 1e-15
PICO = 1e-12
EH = 4.35974417e-18  # Hartrees -> J
EV = 1.6021766209e-19  # eV -> J
H = 6.626069934e-34  # Planck const
KB = 1.38064852e-23  # Boltzmann const
MOLE = 6.02214129e23
KJ = 1000.0
KCAL = 4184.0
# HEADERS
if sys.version[0] == '2':
    STATUS = "STATUS      "
    NEEDINIT = "NEEDINIT    "
    READY = "READY       "
    HAVEDATA = "HAVEDATA    "
    FORCEREADY = "FORCEREADY  "
else:
    STATUS = b"STATUS      "
    NEEDINIT = b"NEEDINIT    "
    READY = b"READY       "
    HAVEDATA = b"HAVEDATA    "
    FORCEREADY = b"FORCEREADY  "
# BYTES
INT = 4
FLOAT = 8


class ExitSignal(BaseException):
    pass


class TimeOutSignal(BaseException):
    pass

class BaseDriver(object):
    """
    Base class of Socket driver.
    """

    def __init__(self, port, addr="127.0.0.1"):
        if iproc == 0:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            '''if sys.version[0]=='2':
                try:
                    self.socket.connect((addr, port))
                    self.socket.settimeout(None)
                except (socket.timeout, ConnectionRefusedError), e:
                    raise TimeOutSignal("Time out, quit.")
            else:'''
            try:
                self.socket.connect((addr, port))
                self.socket.settimeout(None)
            except ConnectionRefusedError:
                #raise ExitSignal("Lost connection")
                sys.exit()
            except socket.timeout:
                raise TimeOutSignal("Time out, quit.")
        self.job_now = 0
        self.job_next = 0
        self.ifInit = False
        self.ifForce = False
        self.cell = None
        self.inverse = None
        self.crd = None
        self.energy = None
        self.force = None
        if sys.version[0]=='2':
            self.extra = ""
        else:
            self.extra = b""
        self.nbead = -1
        self.natom = -1

    def grad(self, crd):
        """
        Calculate gradient.
        Need to be rewritten in inheritance.
        """
        return None, None

    def update(self, text):
        """
        Update system message from INIT motion.
        Need to be rewritten in inheritance.
        Mostly we don't need it.
        """
        pass

    def init(self):
        """
        Deal with message from INIT motion.
        """
        if iproc == 0:
            self.nbead = np.frombuffer(
                self.socket.recv(INT * 1), dtype=np.int32)[0]
            offset = np.frombuffer(self.socket.recv(INT * 1), dtype=np.int32)[0]
            self.update(self.socket.recv(offset))
            self.ifInit = True

    def status(self):
        """
        Reply STATUS.
        """
        if iproc == 0:
            if self.ifInit and not self.ifForce:
                self.socket.send(READY)
            elif self.ifForce:
                self.socket.send(HAVEDATA)
            else:
                self.socket.send(NEEDINIT)

    def posdata(self):
        """
        Read position data.
        """
        if iproc == 0:
            self.cell = np.frombuffer(self.socket.recv(
                FLOAT * 9), dtype=np.float64) * BOHR
            self.inverse = np.frombuffer(self.socket.recv(
                FLOAT * 9), dtype=np.float64) / BOHR
            self.natom = np.frombuffer(
                self.socket.recv(INT * 1), dtype=np.int32)[0]
            crd = np.frombuffer(self.socket.recv(
                FLOAT * 3 * self.natom), dtype=np.float64)
        else:
            crd = None
        crd = comm.bcast(crd, root=0)
        self.crd = crd.reshape((self.natom, 3)) * BOHR
        energy, force = self.grad(self.crd)
        self.energy = energy
        self.force = - force
        self.ifForce = True

    def getforce(self):
        """
        Reply GETFORCE.
        """
        if iproc == 0:
            self.socket.send(FORCEREADY)
            self.socket.send(struct.pack("d", self.energy / EH))
            self.socket.send(struct.pack("i", self.natom))
            for f in self.force.ravel():
                self.socket.send(struct.pack("d", f / (EH / BOHR))
                                )  # Force unit: xx
            try:
                virial = np.diag((self.force * self.crd).sum(axis=0)).ravel() / EH
            except ValueError:
                print(self.atoms, self.force.shape, self.crd.shape, self.crd)
            for v in virial:
                self.socket.send(struct.pack("d", v))
            if len(self.extra) > 0:
                extra = self.extra
            elif sys.version[0] == '2':
                extra = " "
            else:
                extra = b" "
            lextra = len(extra)
            self.socket.send(struct.pack("i", lextra))
            self.socket.send(extra)
            self.ifForce = False

    def exit(self):
        """
        Exit.
        """
        if iproc == 0:
            self.socket.close()
        raise ExitSignal()

    def parse(self):
        """
        Reply the request from server.
        """
        if iproc == 0:
            try:
                self.socket.settimeout(10)
                header = self.socket.recv(12).strip()
                if sys.version[0] == '3':
                    header = header.decode()
                self.socket.settimeout(None)
            except socket.timeout as e:
                raise TimeOutSignal("Time out, quit.")
            if len(header) < 2:
                raise TimeOutSignal()
        else:
            header = None
        header = comm.bcast(header, root=0)
        if header == "STATUS":
            self.status()
        elif header == "INIT":
            self.init()
        elif header == "POSDATA":
            self.posdata()
        elif header == "GETFORCE":
            self.getforce()
        elif header == "EXIT":
            self.exit()


class GaussDriver(BaseDriver):
    """
    Driver for MD calculation with OSV-MP2.
    """
    def __init__(self, port, addr, template, atoms):
        BaseDriver.__init__(self, port, addr)
        with open(template, "r") as f:
            text = f.readlines()
        self.template = ""
        for l in text:
            if 'coord' not in l:
                self.template += l
            else:
                break
        self.atoms = atoms
    def grad(self, crd):
        def read_eg(output):
            with open(output, 'r') as f:
                lines = f.readlines()
                for idx, l in enumerate(lines):
                    if "Sum of electronic and thermal Free Energies=" in l:
                        ene = float(l.split()[-1])
                    elif "Center     Atomic                   Forces" in l:
                        grad = []
                        idx0 = idx+3
                        l_grad = lines[idx0]
                        while "---------" not in l_grad:
                            l_grad = l_grad.split()[-3:]
                            gi = []
                            for num in l_grad:
                                gi.append(float(num))
                            grad.append(gi)
                            idx0 += 1
                            l_grad = lines[idx0]
                        break
            return ene, np.asarray(grad)

        crd = crd/ANGSTROM
        atom = ""
        for i in range(len(self.atoms)):
            atom += self.atoms[i]+'\t'+str(crd[i, 0])+'\t'+str(crd[i, 1])+'\t'+str(crd[i, 2])+'\n'
        with open('1.com', 'w') as f:
            text = self.template + atom + '\n'
            f.write(text)
        #Run gaussian to obtain energy and gradient

        ene, grad = read_eg("1.log")
        return ene*EH, grad*(EH/BOHR)

