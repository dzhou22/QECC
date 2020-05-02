# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit.extensions.standard import C3XGate

#import math tools
import numpy as np
from scipy.linalg import expm

# We import the tools to handle general Graphs
import networkx as nx

# We import plotting tools 
import matplotlib.pyplot as plt 
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter

# Loading your IBM Q account(s)
import account
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-university', group='harvard-lukin', project='phys160')

# Hamming partity check
H = np.matrix([[1, 0, 1, 0, 1, 0, 1],
               [0, 1, 1, 0, 0, 1, 1],
               [0, 0, 0, 1, 1, 1, 1]])

# Hamming generator
G = np.matrix([[1, 0, 1, 0, 1, 0, 1],
               [0, 1, 1, 0, 0, 1, 1],
               [0, 0, 0, 1, 1, 1, 1],
               [1, 1, 1, 0, 0, 0, 0]])

# Noise model

def get_noise(p1, p2, p4):
    '''Returns noie model for given error probabilities for 1, 2, and 4
    qubit gates
    '''
    error_1 = depolarizing_error(p1, 1)
    error_2 = depolarizing_error(p2, 2)
    error_4 = depolarizing_error(p4, 4)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['x', 'h']) 
    noise_model.add_all_qubit_quantum_error(error_2, ['cx']) 
    noise_model.add_all_qubit_quantum_error(error_4, ['c3x']) 
    return noise_model


# Ideal output

def ideal_log0():
    '''Return log0 state as a dictionary of basis elements
    '''
    res = dict()
    for i in range(8):
        to_mult = np.matrix([int(b) for b in list(reversed(list(bin(i)[2:].zfill(3))))])
        basis_elem = np.array(np.matmul(to_mult, H) % 2)[0].tolist()
        dict_key = ''.join([str(b) for b in list(reversed(basis_elem))])
        res[dict_key] = 1 / np.sqrt(8)
    return res

def ideal_log1():
    '''Return log1 state as a dictionary of basis elements
    '''
    res = dict()
    all_ones = np.matrix([1] * 7)
    for i in range(8):
        to_mult = np.matrix([int(b) for b in list(reversed(list(bin(i)[2:].zfill(3))))])
        basis_elem = np.array((np.matmul(to_mult, H) + all_ones) % 2)[0].tolist()
        dict_key = ''.join([str(b) for b in list(reversed(basis_elem))])
        res[dict_key] = 1 / np.sqrt(8)
    return res

# Prepare circuit

def apply_matrix(mat, source, target, circuit):
    '''Apply binary matrix `mat` using CNOT gates with input qubits
    `source` and output qubits `target`. Therefore assertion should
    hold
    '''
    assert (len(target), len(source)) == mat.shape
    for i in range(len(target)):
        for j in range(len(source)):
            if mat[i,j]:
                circuit.cx(source[j], target[i])

def init_log0(circuit, codequbits):
    '''Initialize 7-qubit logical 0, the equally weighted superposition of
    all even elements of the Hamming code. Because the restriction of
    H to columns 1,2,3 has full rank, the superposition is created
    using these qubits
    '''
    circuit.h(cq[1:4]) # equally weighted superposition of 3-dimensional subspace
    basis_change = np.matrix([[1, 1, 0],
                              [1, 0, 0],
                              [0, 0, 1]])
    source = [1, 2, 3]
    target = [0, 4, 5, 6]
    to_apply = np.matmul(H.getT()[target, :], basis_change) % 2
    apply_matrix(to_apply, codequbits[source], codequbits[target], circuit)

def init_log1(circuit, codequbits):
    '''Initialize 7-qubit logical 1
    '''
    init_log0(circuit, codequbits)
    circuit.x(codequbits)

def correct_flips(circuit, codequbits, flipqubits):
    '''Correct flips for the 7-qubit code
    '''
    apply_matrix(H, codequbits, flipqubits, circuit)
    for i in range(1, 8):
        control = list(reversed(list(bin(i)[2:].zfill(3))))
        xposes = [j for j in range(3) if control[j] == '0']
        if xposes:
            circuit.x(flipqubits[xposes])
        circuit.append(C3XGate(), flipqubits[:] + [codequbits[i - 1]], [])
        if xposes:
            circuit.x(flipqubits[xposes])

def correct_phases(circuit, codequbits, phasequbits):
    '''Correct phase errors for the 7-qubit code, which is equivalent to
    correcting flips in a rotated basis
    '''
    circuit.h(codequbits)
    correct_flips(circuit, codequbits, phasequbits)
    circuit.h(codequbits)

def print_results(name, counts, shots):
    # approximate fidelity as probability of perfect outcome
    fidelity = counts.get('0000000', 0) / shots
    print(name + ': ' + 'fidelity=' + str(fidelity))

shots = 2048
cq = QuantumRegister(7, 'code')
aqf = QuantumRegister(3, 'ancillaflip')
aqp = QuantumRegister(3, 'ancillaphase')
cc = ClassicalRegister(7, 'classic')
init_circ = QuantumCircuit(cq, aqf, aqp, cc)
init_log0(init_circ, cq)
deinit_circ = init_circ.inverse()
# init_circ.draw(output='mpl').show()
# deinit_circ.draw(output='mpl').show()

measure_circ = QuantumCircuit(cq, aqf, aqp, cc)
measure_circ.measure(cq, cc)
# measure_circ.draw(output='mpl').show()

# Verify initialization of logical state worked
counts = execute(init_circ + deinit_circ + measure_circ,
                 Aer.get_backend('qasm_simulator'), shots=shots).result().get_counts()
print_results('0 corrections', counts, shots)

flip_circ = QuantumCircuit(cq, aqf, aqp, cc)
correct_flips(flip_circ, cq, aqf)
# flip_circ.draw(output='mpl').show()

phase_circ = QuantumCircuit(cq, aqf, aqp, cc)
correct_phases(phase_circ, cq, aqp)
# phase_circ.draw(output='mpl').show()

# Make sure it can correct 0 errors
counts = execute(init_circ + flip_circ + phase_circ + deinit_circ + measure_circ,
                 Aer.get_backend('qasm_simulator'), shots=shots).result().get_counts()
print_results('1 correction, no errors', counts, shots)

error_circ = QuantumCircuit(cq, aqf, aqp, cc)
error_circ.x(cq[3])
error_circ.z(cq[5])
# error_circ.draw(output='mpl').show()

# Correct 1 flip and 1 phase error
counts = execute(init_circ + error_circ + flip_circ + phase_circ +
                 deinit_circ + measure_circ,
                 Aer.get_backend('qasm_simulator'), shots=shots).result().get_counts()
print_results('1 correction, 1 flip, 1 phase errors', counts, shots)

more_error_circ = QuantumCircuit(cq, aqf, aqp, cc)
more_error_circ.x(cq[[2,3]])
more_error_circ.z(cq[5])
# more_error_circ.draw(output='mpl').show()

# Fail to correct 2 flips and 1 phase error
counts = execute(init_circ + more_error_circ + flip_circ + phase_circ
                 + measure_circ,
                 Aer.get_backend('qasm_simulator'), shots=shots).result().get_counts()
print_results('1 correction, 2 flip, 1 phase errors', counts, shots)

# Run with depolarizing noise
noise_model = get_noise(.001, .001, .001)
counts = execute(init_circ + deinit_circ + measure_circ,
                 Aer.get_backend('qasm_simulator'),
                 noise_model=noise_model, shots=shots).result().get_counts()
print_results('0 corrections, depolarizing errors', counts, shots)
counts = execute(init_circ + flip_circ + phase_circ + deinit_circ +
                 measure_circ, Aer.get_backend('qasm_simulator'),
                 noise_model=noise_model, shots=shots).result().get_counts()
print_results('1 correction, depolarizing errors', counts, shots)

