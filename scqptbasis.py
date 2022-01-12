"""
Basis class for self consistent quantum process tomography
"""
import numpy as np
from typing import Union, Callable, Optional, Dict

# import qiskit classes
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import RXGate, RYGate
from qiskit.quantum_info import PTM


class SelfConsistTomographyBasis:

    def __init__(self,
                 name: str,
                 gates: Dict[str, Union[Callable, Gate]]
                 ):
        self.name = name
        self.labels = list(gates.keys())
        self.gates = gates
        self.matrices = {name: np.real(self._gate_matrix(gate))
                         for (name, gate) in gates.items()}

    @staticmethod
    def _gate_matrix(gate):
        """Gets a PTM representation of the gate"""
        if isinstance(gate, Gate):
            return PTM(gate).data

        if callable(gate):
            c = QuantumCircuit(1)
            gate(c, c.qubits[0])
            return PTM(c).data
        return None

    # adapted from qiskits 'gatesetbasis.py'
    def add_gate(self, gate: Union[Callable, Gate], name: Optional[str] = None):
        """Adds a new gate to the gateset
            Args:
                gate: Either a qiskit gate object or a function taking
                (QuantumCircuit, QuantumRegister)
                and adding the gate to the circuit
                name: the name of the new gate
            Raises:
                RuntimeError: If the gate is given as a function but without
                a name.
        """
        if name is None:
            if isinstance(gate, Gate):
                name = gate.name
            else:
                raise RuntimeError("Gate name is missing")
        self.labels.append(name)
        self.gates[name] = gate
        self.matrices[name] = self._gate_matrix(gate)

    # adapted from qiskits 'gatesetbasis.py'
    def add_gate_to_circuit(self,
                            circ: QuantumCircuit,
                            qubit: QuantumRegister,
                            op: str
                            ):
        """
        Adds the gate op to circ at qubit
        Args:
            circ: the circuit to apply op on
            qubit: qubit to be operated on
            op: gate name
        Raises:
            RuntimeError: if `op` does not describe a gate
        """
        if op not in self.gates:
            raise RuntimeError("{} is not a SPAM circuit".format(op))
        gate = self.gates[op]
        if callable(gate):
            gate(circ, qubit)
        if isinstance(gate, Gate):
            circ.append(gate, [qubit], [])

    def matrix(self, label: str) -> np.ndarray:
        """
        returns the matrix of the Gate with label "label" from the gate set
        """
        return self.matrices[label]


def default_scqpt_basis():
    """
    Returns a default tomographically-complete gateset basis
    Return value: The basis set used in 1211.0322 by S. Merkel et. al.
    """
    gates = {
        'Id': lambda circ, qubit: None,
        'X_Rot_90': lambda circ, qubit: circ.append(RXGate(np.pi/2), [qubit]),
        'Y_Rot_90': lambda circ, qubit: circ.append(RYGate(np.pi/2), [qubit]),
        'X_Rot_180': lambda circ, qubit: circ.append(RXGate(np.pi), [qubit]),
    }
    return SelfConsistTomographyBasis('DefaultSCQPT', gates)
