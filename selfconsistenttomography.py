from typing import Optional, Dict
from time import time

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.quantum_info import PTM

from selfconsist_circuits import sc_process_tomography_circuits
from selfconsist_fitter import SelfConsistentProcessTomographyFitter
from selfconsist_fitter import plot_pauli_transfer_matrix


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
        print("Gateset is: {}".format(gateset.keys()))
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
