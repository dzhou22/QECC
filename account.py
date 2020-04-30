import qiskit
from qiskit import IBMQ
with open('token.txt') as f:
    token = f.readlines()[0]
IBMQ.save_account(token)
