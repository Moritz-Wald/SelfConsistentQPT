# import qiskit classes and methods
import qiskit
import qiskit.quantum_info as qi
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.ignis.verification import process_tomography_circuits, ProcessTomographyFitter
from qiskit.providers.aer import noise
from selfconsistenttomography import self_consistent_tomography

# Setup Circuit
nq = 1
shots = 5000
backend = 'qasm_simulator'

# create circuit
q = QuantumRegister(nq)
circ = QuantumCircuit(q)

'''Insert gate sequence here'''

circ.x(q[0])

# Calculate the target operation
target_operator = qi.Operator(circ)
target_Choi = qi.Choi(target_operator)
target_ptm = qi.PTM(target_operator)

noise_model = noise.NoiseModel()
for qubit in range(1):
    read_err = noise.errors.readout_error.ReadoutError([[0.75, 0.25],
                                                        [0.1, 0.9]])
    noise_model.add_readout_error(read_err, [qubit])

# QPT
print("----------QPT----------")
qpt_circs = process_tomography_circuits(circ, q)
# noinspection PyTypeChecker
execute = qiskit.execute(qpt_circs, qiskit.Aer.get_backend(backend), shots=shots,
                         noise_model=None)
qpt_tomo = ProcessTomographyFitter(execute.result(), qpt_circs)
choi_fit_lstsq = qpt_tomo.fit(method='lstsq')
print('Average gate fidelity: F = {:.10f}'
      .format(qi.average_gate_fidelity(choi_fit_lstsq,
                                       target=target_operator)))
print('Process fidelity: F = {:.10f}'
      .format(qi.process_fidelity(choi_fit_lstsq, target=target_operator,
                                  require_tp=True, require_cp=True)))

# SCQPT
print("----------SCQPT----------")
PTMs = self_consistent_tomography(circ,
                                  shots,
                                  noise_model=None,
                                  backend=backend,
                                  visualize=True,
                                  runtime=True)

# calculate the process and average gate fidelity of the measured Choi
print('Average gate fidelity: F = {:.10f}'
      .format(qi.average_gate_fidelity(PTMs['x'],
                                       target=target_operator)))
print('Process fidelity: F = {:.10f}'
      .format(qi.process_fidelity(PTMs['x'], target=target_ptm,
                                  require_tp=True, require_cp=True)))
