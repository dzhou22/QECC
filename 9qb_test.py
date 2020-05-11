# %matplotlib inline
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.extensions.standard import XGate

#import math tools
import numpy as np
from scipy.linalg import expm
from random import random

# We import the tools to handle general Graphs
import networkx as nx

# We import plotting tools 
import matplotlib.pyplot as plt 
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter
# %config InlineBackend.figure_format = 'svg' # Makes the images look nice

import qiskit.providers.aer.noise as noise


import account

# Loading your IBM Q account(s)
IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q-university', group='harvard-lukin', project='phys160')
provider = IBMQ.get_provider(hub='ibm-q-university', group='harvard-lukin', project='phys160')

class Nine:

    def __init__(self, initial_state = [1,0]):
        self.initial_state = initial_state / np.linalg.norm(initial_state)
        # prepare quantum circuit
        self.qr = QuantumRegister(11)
        self.qc = QuantumCircuit(self.qr)
        self.qc.initialize(self.initial_state, 0)

        self.qc.cx(0, 3)
        self.qc.cx(0, 6)

        for i in range(0, 9, 3):
            self.qc.h(i)
            self.qc.cx(i, i+1)
            self.qc.cx(i, i+2)

    def bitflip_shor9(self, block, correction = True):
        '''
        initial circuit encodes state into logical qubits
        block determines which block of three to correct for (need 1, 2, or 3)
        '''
        c = ClassicalRegister(2)
        self.qc = self.qc + QuantumCircuit(c)
        i = (block-1)*3
        self.qc.cx(i, 9)
        self.qc.cx(i+1, 9)
        self.qc.cx(i, 10)
        self.qc.cx(i+2, 10)
        self.qc.measure([9, 10], [c[0], c[1]])

        # correction based on measured 
        if correction:
            self.qc.x(i+2).c_if(c, 2)
            self.qc.x(i+1).c_if(c, 1)
            self.qc.x(i).c_if(c, 3)

    # function to add phase flip correction portion to circuit
    # last 2 qubits are ancilla
    def phaseflip_shor9(self, correction = True):
        self.qc.barrier()
        c = ClassicalRegister(2)
        self.qc = self.qc + QuantumCircuit(c)
        self.qc.h([i for i in range(9)])
        self.qc.cx([i for i in range(0, 6)], [9 for i in range(0, 6)])
        self.qc.cx([i for i in range(3, 9)], [10 for i in range(3, 9)])
        self.qc.h([i for i in range(9)])
        self.qc.measure([9, 10], [c[0], c[1]])

        # correction based on measured 
        if correction:
            self.qc.z(3).c_if(c, 2)
            self.qc.z(0).c_if(c, 1)

            # self.qc.z(0).c_if(c, 3) #is this right
            # self.qc.z(3).c_if(c, 3) #is this right
            self.qc.z(2).c_if(c, 3)

    def draw(self):
        print(self.qc.draw())

    def add_bitflip(self, p):
        # with probability p, each of the 9 encoded bits will be flipped
        for i in range(9):
            if (random() < p):
                self.qc.x(self.qr[i])

    def add_phaseflip(self, p):
        # with probability p, each of the 9 encoded bits will be flipped
        for i in range(9):
            if (random() < p):
                self.qc.z(self.qr[i])

    def test_correction(self):
        self.qc.barrier()
        cr = ClassicalRegister(9)
        self.qc = self.qc + QuantumCircuit(cr)
        self.qc.measure(range(9), cr)
        emulator = Aer.get_backend('qasm_simulator')
        job = execute(self.qc, emulator, shots=1000)
        hist = job.result().get_counts()
        return hist


def fidelity(desired_counts, actual_counts):
  des_tot = sum(list(desired_counts.values()))
  act_tot = sum(list(actual_counts.values()))
  fidelity = 0

  for key in desired_counts.keys():
    # if actual_counts[key]:
    if key in actual_counts:
      fidelity += np.sqrt(desired_counts[key]*actual_counts[key]*(1/(des_tot*act_tot)))
  return fidelity**2



# Testing

fidelities = []

qc = Nine()
qc.bitflip_shor9(1)
qc.bitflip_shor9(2)
qc.bitflip_shor9(3)
qc.phaseflip_shor9()
desired = qc.test_correction()

# qc2 = Nine()
# qc2.add_bitflip(0.01)
# qc2.add_phaseflip(0.01)
# qc2.bitflip_shor9(1)
# qc2.bitflip_shor9(2)
# qc2.bitflip_shor9(3)
# qc2.phaseflip_shor9()
# actual = qc2.test_correction()

# print(fidelity(desired, actual))

for p in np.arange(0, 0.2, 0.001):
    qc2 = Nine()
    qc2.add_bitflip(p)
    qc2.add_phaseflip(p)
    qc2.bitflip_shor9(1)
    qc2.bitflip_shor9(2)
    qc2.bitflip_shor9(3)
    qc2.phaseflip_shor9()
    actual = qc2.test_correction()

    fidelities.append(fidelity(desired, actual))

plt.plot(np.arange(0, 0.2, 0.001), fidelities)
plt.title("9 qubit code")
plt.ylabel("Fidelity")
plt.xlabel("error probability")
plt.show()
# qc.draw()