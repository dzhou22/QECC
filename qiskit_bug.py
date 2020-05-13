from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error

import numpy as np

def get_noise(p):
    error_1 = depolarizing_error(p, 1)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['x', 'u3']) 
    return noise_model

# 50 x gates
x_circ = QuantumCircuit(1, 1)
for i in range(51):
    x_circ.x(0)
x_circ.measure(0, 0)

# Should be equivalent
u3_circ = QuantumCircuit(1, 1)
for i in range(51):
    u3_circ.u3(np.pi, 0, 0, 0) # Equivalent to x
u3_circ.measure(0, 0)

x_counts = execute(x_circ, Aer.get_backend('qasm_simulator'),
                   noise_model=get_noise(.1)).result().get_counts()
u3_counts = execute(u3_circ, Aer.get_backend('qasm_simulator'),
                   noise_model=get_noise(.1)).result().get_counts()
print('Should be about the same:')
print(x_counts, u3_counts)
# Outputs:
# Should be about the same:
# {'0': 505, '1': 519} {'0': 53, '1': 971}
