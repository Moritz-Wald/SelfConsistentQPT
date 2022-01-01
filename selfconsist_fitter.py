"""
least squares estimation fitter for self consistent quantum process tomography
"""
# python and utility imports
import itertools
import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Optional, Dict

# qiskit imports
from qiskit.ignis.verification.tomography.fitters import TomographyFitter
from qiskit.ignis.verification.tomography.fitters.gateset_fitter import get_cholesky_like_decomposition
from qiskit.result import Result
from qiskit.quantum_info import Choi, PTM


class SelfConsistentProcessTomographyFitter:
    """
    This Fitter is aimed at solving the reconstruction effort for a gate set
    from a in the sense of scqpt informationally complete set of data. A least-
    squares approach is taken.
    """

    def __init__(self,
                 result: Result,
                 circuits: List,
                 gateset: Dict[str, PTM]
                 ):

        # The gate set to characterise
        self.gateset = gateset

        # Add initial data
        self._data = TomographyFitter(result, circuits).data
        self.probs = {}
        for key, vals in self._data.items():
            self.probs[key] = vals.get('0', 0) / sum(vals.values())

    def data(self):
        return self._data.items()

    def fit(self):
        """
        Fit the obtained data to a set of Gates. Use scipy optimize package
        and in that the 'SLSQP' routine, i.e. sequential least squares
        programming.

        :return: the set of Gates in the gate set to optimize in PTM
            representation
        """
        optimizer = ScqptLeastSquaresOptimizer(self.probs, self.gateset)

        optimizer.set_initial_value(self.gateset)
        optimization_results = optimizer.optimize(self.gateset)
        return optimization_results


class ScqptLeastSquaresOptimizer:
    """
    The Optimizer Class to implement the least-squares estimation for the self-
    consistent gateset Fitter.

    The approach taken is similar, but adapted from, to the gate estimation
    technique used for GST in arXiv:1509.02921 or the qiskit implementation of
    GST for that matter.
    DISCLAIMER: The implementation shares the structure with the one in qiskit
    and even some parts of code entirely.
    """

    def __init__(self,
                 probabilities: Dict[Tuple[str], float],
                 gateset: Dict[str, PTM]):

        # set the Pauli Z initial state and measurement in a Form that matmul
        # with PTMs requires -> vectorisation in respect to Pauli matrices
        # E = |0><0|
        self.measurement = np.array([[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]])
        # rho = |0><0|
        self.rho = np.array([[np.sqrt(0.5)], [0], [0], [np.sqrt(0.5)]])

        # Gateset of the form dict{str: PTM}
        self.gateset = gateset
        # Probs of the form dict{Triple[str]: float}
        self.probs = probabilities

        # Data for objective function of form list{Tuple(Triple[str], float)}
        self.obj_fn_data = self._compute_objective_function_data()

        self.initial_value = None

        # set number of qubits at hand
        self.qubits = 1

    def _compute_objective_function_data(self) -> List:
        """
        THIS IS ADAPTED FROM QISKIT API.
        Prepares the data that is needed for the generation of the goal function
        so that calculation in the optimizer is quick and doesnt have to
        recalculate non-changing information.
        Generate every possible combination of Gates as a 3 length product and
        store the corresponding probability p_ijk with it as a tuple
        ((i,j,k), p_ijk).

        :return: Data tuples needed for the calculation of the goal function
        """
        combinations = list(itertools.product(
            list(self.gateset.keys()), repeat=3))
        obj_fn_data = list(zip(combinations, list(self.probs.values())))
        return obj_fn_data

    @staticmethod
    def _complex_matrix_to_vec(M):
        """
        THIS IS TAKEN FROM QISKIT API.
        Turn a complex matrix into its vectorised representation. First come the
        real parts and afterwards the imaginary parts in the vector.

        :param M: The matrix to be vectorized.
        :return: The vectorised vector vec(M.real + M.imag)
        """
        mvec = M.reshape(M.size)
        return list(np.concatenate([mvec.real, mvec.imag]))

    @staticmethod
    def _vec_to_complex_matrix(vec: np.array) -> np.array:
        """
        THIS IS TAKEN FROM QISKIT API.
        Turn a vector with real entries into a complex matrix. First come the
        real parts and afterwards the imaginary parts.

        :param vec: The vector to turn into a complex matrix vec(M.real +
            M.imag).
        :return: The complex matrix corresponding to M
        """
        n = int(np.sqrt(vec.size / 2))
        if 2 * n * n != vec.size:
            raise RuntimeError("Vector of length {} cannot be reshaped"
                               " to square matrix".format(vec.size))
        size = n * n
        return np.reshape(vec[0:size] + 1j * vec[size: 2 * size], (n, n))

    @staticmethod
    def _split_list(input_list: List, sizes: List) -> List[List]:
        """
        THIS IS TAKEN FROM QISKIT API.
        Splits a list to several lists of given size
        Args:
            input_list: A list
            sizes: The sizes of the split lists
        Returns:
            list: The split lists
        Example:
            >> split_list([1,2,3,4,5,6,7], [1,4,2])
            [[1],[2,3,4,5],[6,7]]
        Raises:
            RuntimeError: if length of l does not equal sum of sizes
        """
        if sum(sizes) != len(input_list):
            msg = "Length of list ({}) " \
                  "differs from sum of split sizes ({})".format(len(input_list), sizes)
            raise RuntimeError(msg)
        result = []
        i = 0
        for s in sizes:
            result.append(input_list[i:i + s])
            i = i + s
        return result

    def _join_input_vector(self, gates: List[PTM]) -> np.ndarray:
        """
        THIS IS ADAPTED FROM QISKIT API.
        Generate value vector of parameterization for a set of gates

        :param gates: List of Gates to parameterize
        :return: The parameterized value vector
        """
        # save Choi of Gates in List
        gates_as_choi = [Choi(gate).data for gate in gates]
        # generate T so the gate is Choi = T*T^T
        gates_T = [get_cholesky_like_decomposition(gate) for gate in gates_as_choi]
        result = []
        # rewrite in vector form of first real and then imag parts and append to
        # result vector
        for gate_T in gates_T:
            result += self._complex_matrix_to_vec(gate_T)
        return np.asarray(result)

    def _split_input_vector(self, x: np.ndarray) -> Dict[str, PTM]:
        """
        THIS IS ADAPTED FROM QISKIT API.
        Split the parameterization vector into lists corresponding to one gate
        operation and reconstruct the PTM for each gate

        :param x: the parameterization vector of the gate set
        :return: The set of gates described by the vector at hand. Returns
            Dict{str: PTM}.
        """
        # number of gates
        n = len(self.gateset.keys())

        gateset = {}
        # Hilbert space dimension d = 2 ** N
        dim = 2 ** self.qubits
        # density operator dimension d x d
        densdim = dim ** 2
        # length of parameterization vector of Chi Matrix
        paramdim = 2 * (densdim ** 2)

        # split the value vector in sizes that each corresponds to 1 gate
        vecs = np.asarray(self._split_list(list(x), [paramdim] * n))
        # reshape to complex matrix
        gates_T = [self._vec_to_complex_matrix(vecs[i]) for i in range(n)]
        # cast the gate G=T*T^T to PTM
        gates = [PTM(Choi(np.matmul(cholT, np.conj(cholT.T)))) for cholT in gates_T]
        # match with keys
        for i, key in enumerate(self.gateset.keys()):
            gateset[key] = gates[i]

        return gateset

    def _log_likelihood_func(self, x: np.ndarray) -> float:
        """
        Calculate the value of the log likelihood function given by
        \langle\!\langle\M_0|\R_kR_jR_i|\rho_0\rangle\!\rangle
        as proposed by S. Merkel et al. in eq. 26 in arXiv:1211.032

        :return: function value of the likelihood functional
        """
        current_gateset = self._split_input_vector(x)
        result = 0.
        # for all measurements
        for meas in self.obj_fn_data:
            # initialise ground state
            value = self.rho
            # add all 3 gates to it
            for gate in meas[0]:
                value = np.matmul(current_gateset[gate].data, value)
            # add measurement operator
            value = np.matmul(self.measurement, value)
            # take real part since its a expectation value
            value = np.real(value[0][0])
            # subtract the experimental probability m_ijk
            value = value - meas[1]
            # square for 2 norm
            value = value ** 2
            # add up for sum
            result += value
        return result

    def _bounds_eq_constraint(self, x: np.array) -> List[float]:
        """Equality MLE constraints on the GST data
        Args:
            x: The vector representation of the GST data
        Returns:
            The list of computed constraint values (should equal 0)
        Additional information:
            We have the following constraints on the GST data, due to
            the PTM representation we are using:
            1) G_{0,0} is 1 for every gate G
            2) The rest of the first row of each G is 0.
            3) G only has real values, so imaginary part is 0.
            For additional info, see section 3.5.2 in arXiv:1509.02921
        """
        # get list of PTMs representing the gates
        ptm_matrix = []
        for mat in self._split_input_vector(x).values():
            ptm_matrix = ptm_matrix + self._complex_matrix_to_vec(mat.data)
        bounds_eq = []
        n = len(self.gateset.keys())
        d = (2 ** self.qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2

        i = 0
        for _ in range(n):  # iterate over all Gs
            bounds_eq.append(ptm_matrix[i] - 1)  # G^k_{0,0} is 1
            i += 1
            for _ in range(ds - 1):
                bounds_eq.append(ptm_matrix[i] - 0)  # G^k_{0,i} is 0
                i += 1
            for _ in range((ds - 1) * ds):  # rest of G^k
                i += 1
            for _ in range(ds ** 2):  # the complex part of G^k
                bounds_eq.append(ptm_matrix[i] - 0)  # G^k_{0,i} is 0
                i += 1
        return bounds_eq

    def _bounds_ineq_constraint(self, x: np.array) -> List[float]:
        """Inequality MLE constraints on the GST data
        Args:
            x: The vector representation of the GST data
        Returns:
            The list of computed constraint values (should be >= 0)
        Additional information:
            We have the following constraints on the GST data, due to
            the PTM representation we are using:
            1) Every row of G except the first has entries in [-1,1]
            We implement this as two inequalities per entry.
            For additional info, see section 3.5.2 in arXiv:1509.02921
        """
        # get list of PTMs representing the gates
        ptm_matrix = []
        for mat in self._split_input_vector(x).values():
            ptm_matrix = ptm_matrix + self._complex_matrix_to_vec(mat.data)
        bounds_ineq = []
        n = len(self.gateset.keys())
        d = (2 ** self.qubits)  # rho is dxd and starts at variable d^2
        ds = d ** 2

        i = 0
        for _ in range(n):  # iterate over all Gs
            i += 1
            for _ in range(ds - 1):
                i += 1
            for _ in range((ds - 1) * ds):  # rest of G^k
                bounds_ineq.append(ptm_matrix[i] + 1)  # G_k[i] >= -1
                bounds_ineq.append(-ptm_matrix[i] + 1)  # G_k[i] <= 1
                i += 1
            for _ in range(ds ** 2):  # the complex part of G^k
                i += 1
        return bounds_ineq

    def _constraints(self) -> List[Dict]:
        """Generates the constraints for the MLE optimization
        Returns:
            A list of constraints.
        Additional information:
            Each constraint is a dictionary containing
            type ('eq' for equality == 0, 'ineq' for inequality >= 0)
            and a function generating from the input x the values
            that are being constrained.
        """
        cons = [{'type': 'eq', 'fun': self._bounds_eq_constraint},
                {'type': 'ineq', 'fun': self._bounds_ineq_constraint}]
        return cons

    def set_initial_value(self, initial_value):
        """
        Sets the initial value for the optimizer using a set of gates.
        :param initial_value: The Dict of the gateset the initial value vector
            should be based on
        """
        # make list of gates of type PTM
        gates = [initial_value[label] for label in self.gateset.keys()]
        # save vector of parameterization as initial value for optimizer
        self.initial_value = self._join_input_vector(gates)

    def optimize(self, initial_set: Optional[Dict[str, PTM]] = None) -> Dict:
        """
        THIS IS ADAPTED FROM QISKIT API.
        Execute the minimisation of the least squares likelihood functional

        :param initial_set: The Dict of the gateset the initial value vector
            should be based on
        :return: The dict of gates that minimizes the least squares likelihood
            functional under physicality constraints
        """
        # in case there is a initial set passed as starting point, initialise it
        if initial_set is not None:
            self.set_initial_value(initial_set)

        # minimize the log likelihood function L(G)
        results = minimize(self._log_likelihood_func, self.initial_value,
                           method='SLSQP')
        # refactor to gate PTMs
        resultgates = self._split_input_vector(results.x)

        return resultgates
