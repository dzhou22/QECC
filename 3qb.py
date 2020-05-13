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

class BitFlip:

    def __init__(self, initial_state = [1,0]):
        self.initial_state = initial_state / np.linalg.norm(initial_state)
        self.qr = QuantumRegister(5)
        self.cr = ClassicalRegister(5)
        self.qc = QuantumCircuit(self.qr, self.cr)
        self.qc.initialize(self.initial_state, 0)
        self.qc.cx(self.qr[0], self.qr[1])
        self.qc.cx(self.qr[0], self.qr[2])
        self.results = []

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

    def flip_one(self):
        self.qc.x(self.qr[1])

    def add_error(self, p):
        # with probability p, each of the 3 encoded bits will be flipped
        for i in range(3):
            if (random() < p):
                self.qc.x(self.qr[i])

    def run_simulation(self, shots = 8192, error_prob = 0, target_state = '000'):
        # add noise
        error_1 = noise.depolarizing_error(error_prob, 1)
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ['x','z'])

        # run simulation
        self.qc.measure([0,1,2], [4,3,2])
        emulator = Aer.get_backend('qasm_simulator')
        job = execute(self.qc, emulator, noise_model=noise_model, shots=shots)
        hist = job.result().get_counts()
        # print(hist)

        # print statistics
        successes = 0
        first_three = [n[0:3] for n in list(hist.keys())]
        for i in range(len(first_three)):
            if (first_three[i] == target_state):
                successes += hist.get(list(hist.keys())[i])
        # print("success rate:", successes/shots)
        return successes/shots


    def run_real_device(self, device_name, shots = 8192, target_state = '000'):
        self.qc.measure([0,1,2], [4,3,2])
        device = provider.get_backend(device_name)
        job = execute(self.qc, device, shots=shots)
        hist = job.result().get_counts()
        print(hist)

        # print statistics
        successes = 0
        first_three = [n[0:3] for n in list(hist.keys())]
        for i in range(len(first_three)):
            if (first_three[i] == target_state):
                successes += hist.get(list(hist.keys())[i])
        # print("success rate:", successes/shots)
        return successes/shots

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
        self.qc.measure(self.qr[3], self.cr[0]) # first ancillary qubit's measurement in classical register 0
        self.qc.measure(self.qr[4], self.cr[1]) # second ancillary qubit's measurement in classical register 1
        
        # apply X gate to correct error based on measurement outcomes
        self.qc.x(self.qr[0]).c_if(self.cr, 3) # if the classical register says |11>, or "3", then apply on qb 0
        self.qc.x(self.qr[2]).c_if(self.cr, 2) # |10> means qb 3 has the error
        self.qc.x(self.qr[1]).c_if(self.cr, 1) # |01> means qb 1 has the error

    def test_correction(self):
        self.qc.measure([0,1,2], [4,3,2])
        emulator = Aer.get_backend('qasm_simulator')
        job = execute(self.qc, emulator, shots=1)
        hist = job.result().get_counts()
        result = list(hist.keys())[0][0:3]
        # print(result)
        self.results.append(result)
        
    #removes the last 9 gates (use if the last thing you did was correct_error())
    def remove_correction_gates(self): 
        for _ in range(9):
            self.qc.data.pop(len(self.qc.data)-1)
    
    #removes all but the gates added during initialization
    def reset(self): 
        while (len(self.qc.data) > 3):
            self.qc.data.pop(len(self.qc.data) - 1)

    def get_accuracy(self):
        return self.results.count('000')/len(self.results)

class BitFlip2:

    def __init__(self, initial_state = [1,0]):
        self.initial_state = initial_state / np.linalg.norm(initial_state)
        self.qr = QuantumRegister(3)
        self.cr = ClassicalRegister(1)
        self.qc = QuantumCircuit(self.qr, self.cr)
        self.qc.initialize(self.initial_state, 0)
        self.qc.cx(self.qr[0], self.qr[1])
        self.qc.cx(self.qr[0], self.qr[2])
        self.results = []

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

    def correct_error(self):
        self.qc.cx(self.qr[0], self.qr[1])
        self.qc.cx(self.qr[0], self.qr[2])

        c2x_gate = XGate().control(2)
        self.qc.append(c2x_gate, [2,1,0])

    def add_error(self, p):
        # with probability p, each of the 3 encoded bits will be flipped
        for i in range(3):
            if (random() < p):
                self.qc.x(self.qr[i])

    #removes the last 3 gates (use if the last thing you did was correct_error())
    def remove_correction_gates(self): 
        for _ in range(3):
            self.qc.data.pop(len(self.qc.data)-1)

    #removes all but the gates added during initialization
    def reset(self): 
        while (len(self.qc.data) > 3):
            self.qc.data.pop(len(self.qc.data) - 1)

    def run_simulation(self, shots = 8192, error_prob = 0, target_state = '0'):
        # add noise
        error_1 = noise.depolarizing_error(error_prob, 1)
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ['x','z'])

        # run simulation
        self.qc.measure(0, 0)
        emulator = Aer.get_backend('qasm_simulator')
        job = execute(self.qc, emulator, noise_model=noise_model, shots=shots)
        hist = job.result().get_counts()
        print(hist)

        # print statistics
        # return hist.get(target_state)/shots

    def run_real_device(self, device_name, shots = 8192, target_state = '0'):
        self.qc.measure(0, 0)
        device = provider.get_backend(device_name)
        job = execute(self.qc, device, shots=shots)
        hist = job.result().get_counts()
        # print(hist)

        # print statistics
        return hist.get(target_state)/shots

    def test_correction(self):
        self.qc.measure(0, 0)
        emulator = Aer.get_backend('qasm_simulator')
        job = execute(self.qc, emulator, shots=1)
        hist = job.result().get_counts()
        result = list(hist.keys())[0]
        # print(result)
        self.results.append(result)

    def get_accuracy(self):
        return self.results.count('0')/len(self.results)

class PhaseFlip:

    def __init__(self, initial_state = [1,0]):
        self.initial_state = initial_state / np.linalg.norm(initial_state)
        self.qr = QuantumRegister(5)
        self.cr = ClassicalRegister(5)
        self.qc = QuantumCircuit(self.qr, self.cr)
        self.qc.initialize(self.initial_state, 0)
        self.qc.cx(self.qr[0], self.qr[1])
        self.qc.cx(self.qr[0], self.qr[2])
        self.qc.h(self.qr[0])
        self.qc.h(self.qr[1])
        self.qc.h(self.qr[2])
        self.results = []

    def draw(self):
        print(self.qc.draw())

    def x_gate(self):
        self.qc.z(self.qr[0])
        self.qc.z(self.qr[1])
        self.qc.z(self.qr[2])

    def z_gate(self):
        self.qc.x(self.qr[0])
        self.qc.x(self.qr[1])
        self.qc.x(self.qr[2])

    def run_simulation(self, shots = 8192, target_state = '000'):
        self.qc.measure([0,1,2], [4,3,2])
        emulator = Aer.get_backend('qasm_simulator')
        job = execute(self.qc, emulator, shots=shots)
        hist = job.result().get_counts()
        print(hist)
        # l = list(hist.keys())
  

    def run_real_device(self, device_name, shots = 8192):
        self.qc.measure([0,1,2], [4,3,2])
        device = provider.get_backend(device_name)
        job = execute(self.qc, device, shots=shots)
        hist = job.result().get_counts()
        print(hist)
        # results = list(hist.keys())
        # for r in results:
        #     self.results.append(r[0:3])

    def add_error(self, p):
        # with probability p, each of the 3 encoded bits will be flipped
        for i in range(3):
            if (random() < p):
                self.qc.z(self.qr[i])

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
        self.qc.z(self.qr[0]).c_if(self.cr, 3) # if the classical register says |11>, or "3", then apply on qb 0
        self.qc.z(self.qr[2]).c_if(self.cr, 2) # |10> means qb 2 has the error
        self.qc.z(self.qr[1]).c_if(self.cr, 1) # |01> means qb 3 has the error

        self.qc.h(self.qr[0])
        self.qc.h(self.qr[1])
        self.qc.h(self.qr[2])
    
    def flip_one(self, i):
        self.qc.z(self.qr[i])

    def test_correction(self):
        self.qc.measure([0,1,2], [4,3,2])
        emulator = Aer.get_backend('qasm_simulator')
        job = execute(self.qc, emulator, shots=1)
        hist = job.result().get_counts()
        result = list(hist.keys())[0][0:3]
        # print(result)
        self.results.append(result)
    
    def get_accuracy(self):
        return self.results.count('000')/len(self.results)

    #removes the last 9 gates (use if the last thing you did was correct_error())
    def remove_correction_gates(self): 
        for _ in range(9):
            self.qc.data.pop(len(self.qc.data)-1)
    
    #removes all but the gates added during initialization
    def reset(self): 
        while (len(self.qc.data) > 6):
            self.qc.data.pop(len(self.qc.data) - 1)

class PhaseFlip2:

    def __init__(self, initial_state = [1,0]):
        self.initial_state = initial_state / np.linalg.norm(initial_state)
        self.qr = QuantumRegister(3)
        self.cr = ClassicalRegister(1)
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
        self.qc.z(self.qr[0])
        self.qc.z(self.qr[1])
        self.qc.z(self.qr[2])

    def z_gate(self):
        self.qc.x(self.qr[0])
        self.qc.x(self.qr[1])
        self.qc.x(self.qr[2])

    def correct_error(self):
        self.qc.h(self.qr[0])
        self.qc.h(self.qr[1])
        self.qc.h(self.qr[2])

        self.qc.cx(self.qr[0], self.qr[1])
        self.qc.cx(self.qr[0], self.qr[2])
        c2x_gate = XGate().control(2)
        self.qc.append(c2x_gate, [2,1,0])

    #removes the last 3 gates (use if the last thing you did was correct_error())
    def remove_correction_gates(self): 
        for _ in range(3):
            self.qc.data.pop(len(self.qc.data)-1)

    #removes all but the gates added during initialization
    def reset(self): 
        while (len(self.qc.data) > 3):
            self.qc.data.pop(len(self.qc.data) - 1)

    def run_simulation(self, shots = 8192, error_prob = 0, target_state = '0'):
        # add noise
        error_1 = noise.depolarizing_error(error_prob, 1)
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ['x','z'])

        # run simulation
        self.qc.measure(0, 0)
        emulator = Aer.get_backend('qasm_simulator')
        job = execute(self.qc, emulator, noise_model=noise_model, shots=shots)
        hist = job.result().get_counts()
        # print(hist)

        # print statistics
        return hist.get(target_state)/shots

    def run_real_device(self, device_name, shots = 8192, target_state = '0'):
        self.qc.measure(0, 0)
        device = provider.get_backend(device_name)
        job = execute(self.qc, device, shots=shots)
        hist = job.result().get_counts()
        # print(hist)

        # print statistics
        return hist.get(target_state)/shots

class SingleQubit():

    def __init__(self):
        self.qr = QuantumRegister(1)
        self.cr = ClassicalRegister(1)
        self.qc = QuantumCircuit(self.qr, self.cr)
        self.results = []

    def x_gate(self):
        self.qc.x(self.qr[0])

    def add_bitflip_error(self, p):
        if (random() < p):
            self.qc.x(self.qr[0])
    
    def add_phaseflip_error(self, p):
        if (random() < p):
            self.qc.z(self.qr[0])

    def run_once(self):
        self.qc.measure(0,0)
        emulator = Aer.get_backend('qasm_simulator')
        job = execute(self.qc, emulator, shots=1)
        hist = job.result().get_counts()
        result = list(hist.keys())[0]
        return result
        # print(hist)
        # return hist.get(target_state)/shots

    def get_accuracy(self):
        return self.results.count('0')/len(self.results)

    def draw(self):
        print(self.qc.draw())

def plot_results(results, title):
    plt.plot(np.arange(0, 1, 0.05), results)
    plt.title(title)
    plt.ylabel("Fidelity")
    plt.xlabel("error probability")
    plt.show()

# # Testing BitFlip Code with hand-added errors
# success_rate = []
# for p in np.arange(0, 1, 0.05):
#     qc = BitFlip()
#     # qc = BitFlip2()
#     for i in range(10000):
#         qc.add_error(p)
#         qc.correct_error()
#         qc.test_correction()
#         qc.reset()  
#     success_rate.append(qc.get_accuracy())

# plot_results(success_rate, '3-qubit bit flip code')

# Let's compare to a single qubit with no error correction:
# success_rate_1qb = []
# for p in np.arange(0, 1, 0.05):
#     results = []
#     for i in range(10000):
#         qc = SingleQubit()
#         qc.add_bitflip_error(p)
#         # qc.draw()
#         results.append(qc.run_once())
#     success_rate_1qb.append(results.count('0')/len(results))

# plot_results(success_rate_1qb, "single qubit")
# plt.plot(np.arange(0, 1, 0.05), success_rate, label="with correction (3 qubits)")
# plt.plot(np.arange(0, 1, 0.05), success_rate_1qb, label="no correction (1 qubit)")
# plt.title("3 qubit bit flip code")
# plt.ylabel("Fidelity")
# plt.xlabel("error probability")
# plt.show()

# # Testing PhaseFlip Code with hand-added errors
# success_rate = []
# for p in np.arange(0, 1, 0.05):
#     qc = PhaseFlip([1,0])
#     for i in range(10000):
#         qc.add_error(p)
#         qc.correct_error()
#         qc.test_correction()
#         qc.reset()
#     success_rate.append(qc.get_accuracy())

# plt.plot(np.arange(0, 1, 0.05), success_rate, label="with correction (3 qubits)")
# plt.plot(np.arange(0, 1, 0.05), success_rate_1qb, label="no correction (1 qubit)")
# plt.title("3-qubit phase flip code")
# plt.ylabel("Fidelity")
# plt.xlabel("error probability")
# plt.legend()
# plt.show()


# # Testing BitFlip Code on noisy X Gate/Z Gate
# success_rate = []
# qc = BitFlip([1,0])
# qc.x_gate()
# qc.correct_error()

# for p in np.arange(0, 1, 0.05):
    # success_rate.append(qc.run_simulation(10000, p, '000'))

# plot_results(success_rate, "3-qubit bit flip with noisy Z gate")

# # Trying to test on real machine: this doesn't work because real machine can't do c_if :(
# print("running...")
# print("accuracy:", qc.run_real_device('ibmq_ourense', 8192, '111'))



# # Testing BitFlip2 Code
qc = BitFlip2()
# qc.x_gate()
qc.correct_error()
# qc.draw()

# qc.run_simulation(100,1,'1')

# success_rate = []
# for p in np.arange(0,1,0.05):
#     success_rate.append(qc.run_simulation(10000,p,'1'))

# plot_results(success_rate, "3-qubit bit flip (method 2) with noisy X gate")

# # # Testing on real machine:
# print("running on ourense...")
# print("accuracy:", qc.run_real_device('ibmq_ourense', 8192, '0'))
# ====RESULT of initialization:==== accuracy: 0.9305419921875
# print("accuracy:", qc.run_real_device('ibmq_ourense', 8192, '1'))
# # ====RESULT:==== accuracy: 0.8673095703125
# print("running on essex...")
# print("accuracy:", qc.run_real_device('ibmq_essex', 8192, '0'))
# ====RESULT of initialization:====
# ====RESULT:==== accuracy: 0.8885498046875
# print("running on rome...")
# print("accuracy:", qc.run_real_device('ibmq_rome', 8192, '1'))
# ====RESULT of initialization:==== accuracy: 0.9193115234375
# ====RESULT:==== accuracy: 0.8143310546875
# accuracy: 0.7313232421875 ???
# accuracy: 0.8616943359375 ???
# accuracy: 0.7362060546875 ???
# accuracy: 0.729736328125
# print("accuracy:", qc.run_real_device('ibmq_athens', 8192, '1')) # not found
# ====RESULT:==== 
# print("running on london...")
# print("accuracy:", qc.run_real_device('ibmq_london', 8192, '0')) #error
# ====RESULT of initialization:==== accuracy: 0.900146484375
# ====RESULT:==== accuracy: 0.8560791015625
# accuracy: 0.643798828125???
# accuracy: 0.7236328125???
# accuracy: 0.8243408203125
# print("running on burlington...")
# print("accuracy:", qc.run_real_device('ibmq_burlington', 8192, '0')) # error
# ====RESULT of initialization:==== accuracy: 0.767822265625
# ====RESULT:==== accuracy: 0.7490234375
# print("running on vigo...")
# print("accuracy:", qc.run_real_device('ibmq_vigo', 8192, '0'))
# ====RESULT of initialization:==== accuracy: 0.8607177734375
# ====RESULT:==== accuracy: 0.80419921875

# # Let's compare to a single qubit with no error correction:
# qr = QuantumRegister(1)
# cr = ClassicalRegister(1)
# single_qb = QuantumCircuit(qr, cr)
# # single_qb.x(qr[0])
# single_qb.measure(0, 0)

# simulation:
# success_rate = []
# for p in np.arange(0,1,0.05):
#     error_1 = noise.depolarizing_error(p, 1)
#     noise_model = noise.NoiseModel()
#     noise_model.add_all_qubit_quantum_error(error_1, ['x','z'])

#     single_qb.measure(0, 0)
#     emulator = Aer.get_backend('qasm_simulator')
#     job = execute(single_qb, emulator, noise_model=noise_model, shots=10000)
#     hist = job.result().get_counts()

#     success_rate.append(hist.get('0'))

# plot_results(success_rate, "single qubit X gate no correction")

# real machines:
# device = provider.get_backend('ibmq_ourense')
# device = provider.get_backend('ibmq_essex')
# device = provider.get_backend('ibmq_rome')
# device = provider.get_backend('ibmq_athens')
# device = provider.get_backend('ibmq_london')
# device = provider.get_backend('ibmq_burlington')
# device = provider.get_backend('ibmq_vigo')

# print("running...")
# job = execute(single_qb, device, shots=8192)
# hist = job.result().get_counts()
# print("accuracy:", hist.get('0')/8192)

# ====RESULT initialization (ourense):==== accuracy: 0.9903564453125
# ====RESULT initialization (essex):==== 
# ====RESULT initialization (rome):==== accuracy: 0.9937744140625
# ====RESULT initialization (athens):==== error
# ====RESULT initialization (london):==== accuracy: 0.997314453125
# ====RESULT initialization (burlington):==== accuracy: 0.980712890625
# ====RESULT initialization (vigo):==== accuracy: 0.9862060546875

# ====RESULT for x gate (ourense):==== accuracy: 0.96875
# ====RESULT for x gate (essex):==== accuracy: 0.958740234375
# ====RESULT for x gate (rome):==== accuracy: 0.952880859375
# ====RESULT for x gate (athens):==== error
# ====RESULT for x gate (london):==== accuracy: 0.941162109375
# ====RESULT for x gate (burlington):==== accuracy: 0.9561767578125
# ====RESULT for x gate (vigo):==== accuracy: 0.9698486328125



# Testing PhaseFlip2 Code
# qc = PhaseFlip2()
# qc.x_gate()
# qc.correct_error()
# # print(qc.run_simulation(10,0,'1'))

# success_rate = []
# for p in np.arange(0,1,0.05):
#     success_rate.append(qc.run_simulation(10000,p,'1'))

# plot_results(success_rate, "3-qubit phase flip (method 2) with noisy X gate")

# real machine:
# # Testing on real machine:
# print("running...")
# print("accuracy:", qc.run_real_device('ibmq_ourense', 8192, '1'))
# ====RESULT:==== accuracy: 0.8614501953125
# print("accuracy:", qc.run_real_device('ibmq_essex', 8192, '1'))
# ====RESULT:==== accuracy: 0.8885498046875
# print("accuracy:", qc.run_real_device('ibmq_rome', 8192, '1'))
# ====RESULT:==== accuracy: 0.8568115234375
# print("accuracy:", qc.run_real_device('ibmq_athens', 8192, '1')) # not found
# ====RESULT:==== 
# print("accuracy:", qc.run_real_device('ibmq_london', 8192, '1'))
# ====RESULT:==== accuracy: 0.8560791015625
# print("accuracy:", qc.run_real_device('ibmq_burlington', 8192, '1')) # error
# ====RESULT:==== 
# print("accuracy:", qc.run_real_device('ibmq_vigo', 8192, '1'))
# ====RESULT:==== accuracy: 0.8740234375