# %matplotlib inline
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *

#import math tools
import numpy as np
from scipy.linalg import expm

# We import the tools to handle general Graphs
import networkx as nx

# We import plotting tools 
import matplotlib.pyplot as plt 
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter
# %config InlineBackend.figure_format = 'svg' # Makes the images look nice

import account

# Loading your IBM Q account(s)
IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q-university', group='harvard-lukin', project='phys160')
IBMQ.get_provider(hub='ibm-q-university', group='harvard-lukin', project='phys160')

class BitFlip:

    def __init__(self, initial_state = [1,0]):
        self.initial_state = initial_state / np.linalg.norm(initial_state)
        self.qr = QuantumRegister(5)
        self.cr = ClassicalRegister(2)
        self.qc = QuantumCircuit(self.qr, self.cr)
        self.qc.initialize(self.initial_state, 0)
        self.qc.cx(self.qr[0], self.qr[1])
        self.qc.cx(self.qr[0], self.qr[2])

    def draw(self):
        print(self.qc.draw())

    def x_gate(self):
        self.qc.x(self.qr[0])
        self.qc.x(self.qr[1])
        self.qc.x(self.qr[2])

    def z_gate(self):
        self.qc.z(self.qr[0])
        self.qc.z(self.qr[1])
        self.qc.z(self.qr[2])

    def run_simulation(self, shots = 8192):
        emulator = Aer.get_backend('qasm_simulator')
        job = execute(self.qc, emulator, shots=shots)
        hist = job.result().get_counts()
        print(hist)
        print(plot_histogram(hist))

    # after performing an operation on the encoded state, correct the error
    def correct_error(self):
        # qr = qc.qregs[0]
        # cr = qc.cregs[0]
        
        # initialize the ancillary qubits to 0
        self.qc.initialize([1,0], 3)
        self.qc.initialize([1,0], 4)
        
        # attach CNOTs
        self.qc.cx(self.qr[0], self.qr[3])
        self.qc.cx(self.qr[1], self.qr[3])
        
        self.qc.cx(self.qr[0], self.qr[4])
        self.qc.cx(self.qr[2], self.qr[4])
        
        # measure the ancillary qubits
        self.qc.measure(self.qr[3], self.cr[0]) # first ancillary qubit's measurement in first classical register
        self.qc.measure(self.qr[4], self.cr[1]) # second ancillary qubit's measurement in second classical register
        
        # apply X gate to correct error based on measurement outcomes
        self.qc.x(self.qr[0]).c_if(self.cr, 3) # if the classical register says |11>, or "3", then apply on qb 0
        self.qc.x(self.qr[1]).c_if(self.cr, 2) # |10> means qb 2 has the error
        self.qc.x(self.qr[2]).c_if(self.cr, 1) # |01> means qb 3 has the error
        
    #removes the last 9 gates (use if the last thing you did was correct_error())
    def remove_correction_gates(self): 
        for _ in range(9):
            self.qc.data.pop(len(self.qc.data)-1)
    
    #removes all but the gates added during initialization
    def reset(self): 
        while (len(self.qc.data) > 3):
            self.qc.data.pop(len(self.qc.data) - 1)

class PhaseFlip:

    def __init__(self, initial_state = [1,0]):
        self.initial_state = initial_state / np.linalg.norm(initial_state)
        self.qr = QuantumRegister(5)
        self.cr = ClassicalRegister(2)
        self.qc = QuantumCircuit(self.qr, self.cr)
        self.qc.initialize(self.initial_state, 0)
        self.qc.cx(self.qr[0], self.qr[1])
        self.qc.cx(self.qr[0], self.qr[2])
        self.qc.h(self.qr[0])
        self.qc.h(self.qr[1])
        self.qc.h(self.qr[2])

    def draw(self):
        print(self.qc.draw())

    def x_gate(self):
        self.qc.x(self.qr[0])
        self.qc.x(self.qr[1])
        self.qc.x(self.qr[2])

    def z_gate(self):
        self.qc.z(self.qr[0])
        self.qc.z(self.qr[1])
        self.qc.z(self.qr[2])

    def run_simulation(self, shots = 8192):
        emulator = Aer.get_backend('qasm_simulator')
        job = execute(self.qc, emulator, shots=shots)
        hist = job.result().get_counts()
        print(hist)
        print(plot_histogram(hist))

    # after performing an operation on the encoded state, correct the error
    def correct_error(self):
        # qr = qc.qregs[0]
        # cr = qc.cregs[0]
        
        # initialize the ancillary qubits to 0
        self.qc.initialize([1,0], 3)
        self.qc.initialize([1,0], 4)
        
        # attach CNOTs with Hadamards
        self.qc.h(self.qr[0])
        self.qc.cx(self.qr[0], self.qr[3])
        self.qc.h(self.qr[0])

        self.qc.h(self.qr[1])
        self.qc.cx(self.qr[1], self.qr[3])
        self.qc.h(self.qr[1])
        
        self.qc.h(self.qr[0])
        self.qc.cx(self.qr[0], self.qr[4])
        self.qc.h(self.qr[0])

        self.qc.h(self.qr[2])
        self.qc.cx(self.qr[2], self.qr[4])
        self.qc.h(self.qr[2])
        
        # measure the ancillary qubits
        self.qc.measure(self.qr[3], self.cr[0]) # first ancillary qubit's measurement in first classical register
        self.qc.measure(self.qr[4], self.cr[1]) # second ancillary qubit's measurement in second classical register
        
        # apply X gate to correct error based on measurement outcomes
        self.qc.x(self.qr[0]).c_if(self.cr, 3) # if the classical register says |11>, or "3", then apply on qb 0
        self.qc.x(self.qr[1]).c_if(self.cr, 2) # |10> means qb 2 has the error
        self.qc.x(self.qr[2]).c_if(self.cr, 1) # |01> means qb 3 has the error
        
    #removes the last 9 gates (use if the last thing you did was correct_error())
    def remove_correction_gates(self): 
        for _ in range(9):
            self.qc.data.pop(len(self.qc.data)-1)
    
    #removes all but the gates added during initialization
    def reset(self): 
        while (len(self.qc.data) > 3):
            self.qc.data.pop(len(self.qc.data) - 1)


qc = BitFlip([1,0])
# qc.x_gate()
# qc.draw()
qc.run_simulation()
# qc.correct_error()
# qc.draw()
# qc.remove_correction_gates()
# qc.draw()
# qc.reset()
# qc.draw()