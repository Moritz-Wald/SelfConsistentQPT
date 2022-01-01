# import qiskit classes and methods
import qiskit.quantum_info as qi
from qiskit import QuantumRegister, QuantumCircuit, Aer
from qiskit.providers.aer import noise

from selfconsistenttomography import self_consistent_tomography

# Setup Circuit
nq = 1
shots = 10000
backend = 'qasm_simulator'

# create circuit
q = QuantumRegister(nq)
circ = QuantumCircuit(q)

'''Insert gate sequence here'''

circ.x(q[0])
# circ.s(q[0])
# circ.h(q[0])


# Calculate the target operation
target_operator = qi.Operator(circ)
target_choi = qi.Choi(target_operator)
target_ptm = qi.PTM(target_choi)

noise_model = noise.NoiseModel()
for qubit in range(1):
    read_err = noise.errors.readout_error.ReadoutError([[0.75, 0.25],
                                                        [0.1, 0.9]])
    noise_model.add_readout_error(read_err, [qubit])

PTMs = self_consistent_tomography(circ,
                                  shots,
                                  noise_model=noise_model,
                                  backend=backend,
                                  linearizeError=False,
                                  visualize=True,
                                  runtime=True)

# calculate the process and average gate fidelity of the measured Choi
print('Average gate fidelity: F = {:.6f}'
      .format(qi.average_gate_fidelity(PTMs['x'], target=target_operator)))
print('Process fidelity: F = {:.6f}'
      .format(qi.process_fidelity(PTMs['x'], target=target_ptm,
                                  require_tp=True, require_cp=True)))
