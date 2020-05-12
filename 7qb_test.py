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

def get_noise(p1, p2, p4, pu3=0):
    '''Returns noise model for given error probabilities for 1, 2, and 4
    qubit gates, as well as a separate probability for u3 gates
    '''
    error_1 = depolarizing_error(p1, 1)
    error_2 = depolarizing_error(p2, 2)
    error_4 = depolarizing_error(p4, 4)
    error_u3 = depolarizing_error(pu3, 1)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['x', 'h']) 
    noise_model.add_all_qubit_quantum_error(error_2, ['cx']) 
    noise_model.add_all_qubit_quantum_error(error_4, ['c3x']) 
    noise_model.add_all_qubit_quantum_error(error_u3, ['u3']) 
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

def init_arbitrary(circuit, codequbits):
    '''Initialize 7-qubit logical state to the physical state in
    codequbits[0], assuming all other qubits are set to 0
    '''
    apply_matrix(np.matrix([[0, 0, 0, 0, 1, 1]]).getT(),
                 codequbits[[0]], codequbits[1:], circuit)
    init_log0(circuit, codequbits)
    
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

def test_circs():
    # Verify initialization of logical state worked
    counts = execute(init0_circ + deinit0_circ + measure_circ,
                     Aer.get_backend('qasm_simulator'), shots=shots).result().get_counts()
    print_results('0 corrections', counts, shots)

    # Make sure it can correct 0 errors
    counts = execute(init0_circ + flip_circ + phase_circ + deinit0_circ + measure_circ,
                     Aer.get_backend('qasm_simulator'), shots=shots).result().get_counts()
    print_results('1 correction, no errors', counts, shots)

    # Correct 1 flip and 1 phase error
    error_circ = QuantumCircuit(cq, aqf, aqp, cc)
    error_circ.x(cq[3])
    error_circ.z(cq[5])
    # error_circ.draw(output='mpl').show()
    counts = execute(init0_circ + error_circ + flip_circ + phase_circ +
                     deinit0_circ + measure_circ,
                     Aer.get_backend('qasm_simulator'), shots=shots).result().get_counts()
    print_results('1 correction, 1 flip, 1 phase errors', counts, shots)

    # Fail to correct 2 flips and 1 phase error
    more_error_circ = QuantumCircuit(cq, aqf, aqp, cc)
    more_error_circ.x(cq[[2, 3]])
    more_error_circ.z(cq[[5]])
    # more_error_circ.draw(output='mpl').show()
    counts = execute(init0_circ + more_error_circ + flip_circ + phase_circ
                     + measure_circ,
                     Aer.get_backend('qasm_simulator'), shots=shots).result().get_counts()
    print_results('1 correction, 2 flip, 1 phase errors', counts, shots)

    # Run with depolarizing noise
    noise_model = get_noise(.001, .001, .001)
    counts = execute(init0_circ + deinit0_circ + measure_circ,
                     Aer.get_backend('qasm_simulator'),
                     noise_model=noise_model, shots=shots).result().get_counts()
    print_results('0 corrections, depolarizing errors', counts, shots)
    counts = execute(init0_circ + flip_circ + phase_circ + deinit0_circ +
                     measure_circ, Aer.get_backend('qasm_simulator'),
                     noise_model=noise_model, shots=shots).result().get_counts()
    print_results('1 correction, depolarizing errors', counts, shots)

    # Run with rotation error
    roterror_circ = QuantumCircuit(cq, aqf, aqp, cc)
    roterror_circ.u3(.3, .3, .3, cq)
    # roterror_circ.draw(output='mpl').show()
    counts = execute(init0_circ + roterror_circ + deinit0_circ +
                     measure_circ, Aer.get_backend('qasm_simulator'),
                     shots=shots).result().get_counts()
    print_results('0 corrections, rotation error', counts, shots)
    counts = execute(init0_circ + roterror_circ + flip_circ + phase_circ +
                     deinit0_circ + measure_circ,
                     Aer.get_backend('qasm_simulator'), shots=shots).result().get_counts()
    print_results('1 correction, rotation error', counts, shots)

def get_fidelities(noise_model):
    '''Compute fidelity of a single qubit x gate with and without error
    correction in the presence of depolarizing noie. The x gate is
    implemented with u3 to isolate it in the noise model.
    '''
    # Base implementation, on single qubit with no encoding
    base_circ = QuantumCircuit(1, 1)
    base_circ.u3(np.pi, 0, 0, 0)
    base_circ.measure(0, 0)
    counts = execute(base_circ, Aer.get_backend('qasm_simulator'),
                     noise_model=noise_model, shots=shots).result().get_counts()
    base_fidelity = counts.get('1', 0) / shots
    
    # Implementation using 7-qubit code
    xgate_circ = QuantumCircuit(cq, aqf, aqp, cc)
    xgate_circ.u3(np.pi, 0, 0, cq)

    # Without correction
    # counts = execute(init0_circ + xgate_circ + deinit1_circ +
    #                  measure_circ, Aer.get_backend('qasm_simulator'),
    #                  noise_model=noise_model, shots=shots).result().get_counts()
    # nocorr_fidelity = counts.get('0000000', 0) / shots

    # With correction
    counts = execute(init0_circ + xgate_circ + flip_circ + phase_circ
                     + deinit1_circ + measure_circ,
                     Aer.get_backend('qasm_simulator'),
                     noise_model=noise_model, shots=shots).result().get_counts()
    corr_fidelity = counts.get('0000000', 0) / shots

    # print('Single qubit x gate: fidelity=' + str(base_fidelity)) 
    # print('Encoded x gate, no correction: fidelity=' + str(nocorr_fidelity)) 
    # print('Encoded x gate, with correction: fidelity=' + str(corr_fidelity))

    return (base_fidelity, corr_fidelity)
    
def predict_fidelities(p):
    '''Rough prediction of fidelities without and with error correction
    based on depolarizing parameter `p` for a single qubit gate. These
    predictions assume the gates in the error correcting procedures
    have no noise.  In qiskit, depolarizing channel with parameter p
    causes no error with probability 1-p+p/4, and causes a flip,
    phase, or flip+phase error each with probability p/4.
    '''
    # Probability that no flip error occurs is (1-p+p/4) + (p/4) by
    # above.
    base_fidelity = 1 - p/2

    # Approximate fidelity that correction succeeds as probability
    # that at most 1 flip error and 1 phase error occurs (strictly
    # speaking this gives a lower bound on fidelity).
    corr_fidelity = ((1 - p + p/4)**7 # no error
                     + 7 * (1 - p + p/4)**6 * (p/4) # 1 flip
                     + 7 * (1 - p + p/4)**6 * (p/4) # 1 phase
                     + 7 * (1 - p + p/4)**6 * (p/4) # 1 flip+phase same qubit
                     + 7 * 6 * (1 - p + p/4)**5 * (p/4) * (p/4)) # 1 phase, 1 flip, different qubits

    return (base_fidelity, corr_fidelity)

def plot_fidelities_noisefree_corr():
    '''Plot fidelities for the case where there is no noise during error
    correction
    '''
    empbase = []
    empcorr = []
    anabase = []
    anacorr = []
    pvals = [i / 100 for i in range(0, 16)]
    for p in pvals:
        fid_noise_model = get_noise(0, 0, 0, p)
        empfids = get_fidelities(fid_noise_model)
        anafids = predict_fidelities(p)
        empbase.append(empfids[0])
        empcorr.append(empfids[1])
        anabase.append(anafids[0])
        anacorr.append(anafids[1])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(pvals, empbase, marker='^', label='No error correction, empirical')
    ax.plot(pvals, empcorr, marker='v', label='With error correction, empirical')
    ax.plot(pvals, anabase, ls='--', label='No error correction, analytical')
    ax.plot(pvals, anacorr, ls=':', label='With error correction, analytical (lower bound)')

    ax.set_xlabel('Depolarization probability')
    ax.set_ylabel('Fidelity of X gate')

    plt.legend(loc=3)
    plt.savefig('noisefree_corr.png')

def plot_fidelities_noisy_corr():
    '''Plot fidelities where all gates have the same depolarization
    parameter. Note that a more realistic model would make multi-qubit
    gates more noisy.
    '''
    empbase = []
    empcorr = []
    pvals = [i / 10000 for i in range(0, 16)]
    for p in pvals:
        fid_noise_model = get_noise(p, p, p, p)
        empfids = get_fidelities(fid_noise_model)
        empbase.append(empfids[0])
        empcorr.append(empfids[1])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(pvals, empbase, marker='^', label='No error correction')
    ax.plot(pvals, empcorr, marker='v', label='With error correction')

    ax.set_xlabel('Depolarization probability')
    ax.set_ylabel('Fidelity of X gate')

    plt.legend(loc=3)
    plt.savefig('noisy_corr.png')

shots = 2048
cq = QuantumRegister(7, 'code')
aqf = QuantumRegister(3, 'ancillaflip')
aqp = QuantumRegister(3, 'ancillaphase')
cc = ClassicalRegister(7, 'classic')

init0_circ = QuantumCircuit(cq, aqf, aqp, cc)
init_arbitrary(init0_circ, cq)
deinit0_circ = init0_circ.inverse()
# init0_circ.draw(output='mpl').show()
# deinit0_circ.draw(output='mpl').show()
init1_circ = QuantumCircuit(cq, aqf, aqp, cc)
init1_circ.x(cq[0])
init_arbitrary(init1_circ, cq)
deinit1_circ = init1_circ.inverse()
# init1_circ.draw(output='mpl').show()
# deinit1_circ.draw(output='mpl').show()

measure_circ = QuantumCircuit(cq, aqf, aqp, cc)
measure_circ.measure(cq, cc)
# measure_circ.draw(output='mpl').show()

flip_circ = QuantumCircuit(cq, aqf, aqp, cc)
correct_flips(flip_circ, cq, aqf)
# flip_circ.draw(output='mpl').show()

phase_circ = QuantumCircuit(cq, aqf, aqp, cc)
correct_phases(phase_circ, cq, aqp)
# phase_circ.draw(output='mpl').show()

# test_circs()

plot_fidelities_noisefree_corr()
plot_fidelities_noisy_corr()
