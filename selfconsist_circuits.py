"""
circuit generator for self consistent quantum process tomography
"""
from typing import List, Union, Tuple, Dict, Any
import itertools
from qiskit import QuantumCircuit
from qiskit import QiskitError
from qiskit.circuit import Gate
from qiskit.quantum_info import PTM

from scqptbasis import default_scqpt_basis, SelfConsistTomographyBasis


def sc_process_tomography_circuits(circuit: QuantumCircuit,
                                   measured_qubits=None,
                                   scqpt_basis: Union[str, SelfConsistTomographyBasis] = 'default'
                                   ) -> Tuple[List[QuantumCircuit], Dict[str, Union[Any, Gate]]]:
    """
    Generate the circuits needed to obtain all data required for self consistent
    quantum process tomography. For a self consistent set of N gates G that at
    least contains the gates to produce a informationally complete system, i.e.
    there can be created a complete set of states and measurements using the
    ground state and the pauli z POVM.

    The gate set {G_0, G_1, ..., G_N} of Gates represented by their PTMs R_{i}
    needs N^3 circuits of the form

    .. math::
            m_{ijk} &= \langle\!\langle E|R_k R_j R_i|\rho\rangle\!\rangle

    to be characterised.

    :param circuit: the quantum circuit to be tomographized
    :param measured_qubits: the qubits to apply tomography to
    :param scqpt_basis: the minimal gate set G to obtain a complete set of
        states and measurements
    :return: A list of qiskit QuantumCircuit objects to obtain informationally
    complete results
    """
    # TODO: Add the implementation for more than one qubit

    # 1 + Maximum of list indices given
    if measured_qubits is None:
        measured_qubits = [0]

    if len(measured_qubits) > 1:
        raise QiskitError("Only 1-qubit scqpt so far")

    # calculate number of qubits at hand
    num_qubits = 1 + max(measured_qubits)

    if scqpt_basis == 'default':
        scqpt_basis = default_scqpt_basis()

    # add gates from circuit to the pool of gates
    for gate in circuit.data:
        scqpt_basis.add_gate(gate[0])

    # generate all combinations of the keys in the Gateset
    combinations = list(itertools.product(list(scqpt_basis.gates.keys()), repeat=3))

    # generate a circuit for every combination and add it to all_circuits
    all_circuits = []
    for (h, k, l) in combinations:
        circ = QuantumCircuit(num_qubits, num_qubits)
        qubit = circ.qubits[measured_qubits[0]]

        # add the three gates to the new circuit in the qubit to measure
        scqpt_basis.add_gate_to_circuit(circ, qubit, h)
        circ.barrier(0)
        scqpt_basis.add_gate_to_circuit(circ, qubit, k)
        circ.barrier(0)
        scqpt_basis.add_gate_to_circuit(circ, qubit, l)

        # add the measurement of the line
        circ.measure(0, 0)

        # add the executed gates as the name of the circuit
        circ.name = str((h, k, l))

        all_circuits.append(circ)

    for i in scqpt_basis.gates.keys():
        scqpt_basis.gates[i] = PTM(scqpt_basis.matrix(i))

    return all_circuits, scqpt_basis.gates
