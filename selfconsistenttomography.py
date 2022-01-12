import itertools as it
from typing import Optional, Dict
from time import time

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.quantum_info import PTM

from selfconsist_circuits import sc_process_tomography_circuits
from selfconsist_fitter import SelfConsistentProcessTomographyFitter


def self_consistent_tomography(circ: QuantumCircuit,
                               shots: int,
                               noise_model: Optional[NoiseModel] = None,
                               backend:  Optional[str] = 'qasm_simulator',
                               visualize: Optional[bool] = False,
                               runtime: Optional[bool] = False
                               ) -> Dict[str, PTM]:
    """
    Function to run a complete tomography using the self consistent quantum
    process tomography approach of the gates in a given circuit.

    :param circ: The circuit to be tomographized. The gates reconstructed are
        taken from this QuantumCircuit Object and added to the pool of gates to
        reconstruct.
    :param shots: The number of experiment runs to observe the frequencies on
        for a given configuration (gate sequence). For lower numbers, the
        resulting probabilities have increasingly significant statistical
        sampling error. To reconstruct the analytical probabilities sufficiently
        well, big numbers are needed here.
    :param noise_model: (default: None) The noise Model object to apply to the
        data when run in the below mentioned backend.
    :param backend: (default: 'qasm_simulator') The backend to execute the
        different circuits on. This is the explizit description of the system
        at hand.
    :param linearizeError: (default: False) If 'True', reconstructs the gateset
        using the linearized least squares function proposed by S. Merkel et al.
        in eq. 29 of arXiv:1211.0322 instead of the non linearized in eq. 26.
    :param visualize: (default: False) If 'True', visualises the Pauli transfer
        matrix representation of the ideal and reconstructed gates side by side
        for each gate in the gate set to be tomographized.
    :param runtime: (default: False) If 'True', prints the runtime of each step
        in the reconstruction. Also prints the number of circuits to be run to
        obtain the informationally complete dataset.
    :return: A Dict instance of the Form {str: PTM} of the gates that have been
        reconstructed for each label in Liouville-Pauli (PTM) representation.
    """

    # load the execution backend
    execbackend = Aer.get_backend(backend)

    # generate the circuits needed for the tomography
    t1 = time()
    scqpt_circuits, gateset = sc_process_tomography_circuits(circ)
    if runtime:
        # print("There are {} circuits being executed on the gate "
        #       "set {}.".format(len(scqpt_circuits), gateset))
        print("Time taken for circuit generation: {:0.3f}".format(time()-t1))

    # run the circuits as an experiment on execbackend and store results
    t2 = time()
    execresults = execbackend.run(scqpt_circuits, shots=shots,
                                  noise_model=noise_model).result()
    if runtime:
        print("Time taken for experiment execution: {:0.3f}".format(time()-t2))

    # create a instance of the SelfConsistentProcessTomographyFitter Class and
    # fit the experimental data to a gateset and store as Dict
    t3 = time()
    Fitter = SelfConsistentProcessTomographyFitter(execresults,
                                                   scqpt_circuits,
                                                   gateset)
    resultPTMs = Fitter.fit()
    if runtime:
        print("Time taken for fitting process: {:0.3f}".format(time()-t3))

    # visualize the PTM representation of the achieved gateset in comparison to
    # the ideal gateset
    if visualize:
        idealPTMs = [g.data for g in gateset.values()]
        expPTMs = [g.data for g in resultPTMs.values()]

        for i, gate in enumerate(gateset.keys()):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            plot_pauli_transfer_matrix(idealPTMs[i], ax1,
                                       title='Ideal of {}'.format(gate))
            plot_pauli_transfer_matrix(expPTMs[i], ax2,
                                       title='Estimate of {}'.format(gate))
            plt.show()

    return resultPTMs


# taken from forest.benchmarking
# https://github.com/rigetti/forest-benchmarking/blob/master/forest/...
# ...benchmarking/plotting/state_process.py
def plot_pauli_transfer_matrix(ptransfermatrix: np.ndarray,
                               ax,
                               labels=None, title='',
                               fontsizes: int = 16):
    """
    Visualize a quantum process using the Pauli-Liouville representation (aka
    the Pauli Transfer Matrix) of the process.

    :param ptransfermatrix: The Pauli Transfer Matrix
    :param ax: The matplotlib axes.
    :param labels: The labels for the operator basis states.
    :param title: The title for the plot
    :param fontsizes: Font size for axis labels
    :return: The modified axis object.
    :rtype: AxesSubplot
    """
    ptransfermatrix = np.real_if_close(ptransfermatrix)
    im = ax.imshow(ptransfermatrix, interpolation="nearest", cmap="RdBu",
                   vmin=-1, vmax=1)
    if labels is None:
        dim_squared = ptransfermatrix.shape[0]
        num_qubits = np.int64(np.log2(np.sqrt(dim_squared)))
        labels = [''.join(x) for x in it.product('IXYZ', repeat=num_qubits)]
    else:
        dim_squared = len(labels)

    cb = plt.colorbar(im, ax=ax, ticks=[-1, -3 / 4, -1 / 2, -1 / 4, 0, 1 / 4,
                                        1 / 2, 3 / 4, 1])
    ticklabs = cb.ax.get_yticklabels()
    cb.ax.set_yticklabels(ticklabs, ha='right')
    cb.ax.yaxis.set_tick_params(pad=35)
    cb.draw_all()
    ax.set_xticks(range(dim_squared))
    ax.set_xlabel("Input Pauli Operator", fontsize=fontsizes)
    ax.set_yticks(range(dim_squared))
    ax.set_ylabel("Output Pauli Operator", fontsize=fontsizes)
    ax.set_title(title, fontsize=int(np.floor(1.2 * fontsizes)), pad=15)
    ax.set_xticklabels(labels, rotation=45,
                       fontsize=int(np.floor(0.7 * fontsizes)))
    ax.set_yticklabels(labels, fontsize=int(np.floor(0.7 * fontsizes)))
    ax.grid(False)
    return ax
