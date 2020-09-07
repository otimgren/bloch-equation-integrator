"""
Functions for integrating Optical Bloch Equations using the method outlined in
John Barry's thesis, section 3.4
"""

import sys
import numpy as np
sys.path.append('../molecular-state-classes-and-functions/')
from classes import UncoupledBasisState, CoupledBasisState, State
from tqdm.notebook import tqdm

def generate_sharp_superoperator(M):
    """
    Given an operator M in Hilbert space, generates sharp superoperator M_L in Liouville space (see "Optically pumped atoms" by Happer, Jau and Walker)

    inputs:
    M = matrix representation of operator in Hilbert space

    outputs:
    M_L = representation of M in in Liouville space
    """
    M_L = np.kron(M.T, np.eye(M.shape[0]))

    return M_L

def generate_flat_superoperator(M):
    """
    Given an operator M in Hilbert space, generates flat superoperator M_L in Liouville space (see "Optically pumped atoms" by Happer, Jau and Walker)

    inputs:
    M = matrix representation of operator in Hilbert space

    outputs:
    M_L = representation of M in in Liouville space
    """
    M_L = np.kron(np.eye(M.shape[0]), M)

    return M_L

def generate_commutator_superoperator(M):
    """
    Function that generates the commutator superoperator in Liouville space

    inputs:
    M = matrix representation of operator in Hilbert space that whose commutator with density matrix is being generated

    outputs:
    M_L = representation of commutator with M in in Liouville space
    """
    M_com = generate_flat_superoperator(M)  - generate_sharp_superoperator(M)

    return M_com


def reduced_basis_hamiltonian(basis_ori, H_ori, basis_red):
    """
    Function that outputs Hamiltonian for a sub-basis of the original basis

    inputs:
    basis_ori = original basis (list of states)
    H_ori = Hamiltonian in original basis
    basis_red = sub-basis of original basis (list of states)

    outputs:
    H_red = Hamiltonian in sub-basis
    """

    #Determine the indices of each of the reduced basis states
    index_red = np.zeros(len(basis_red), dtype = int)
    for i, state_red in enumerate(basis_red):
        index_red[i] = basis_ori.index(state_red)

    #Initialize matrix for Hamiltonian in reduced basis
    H_red = np.zeros((len(basis_red),len(basis_red)), dtype = complex)

    #Loop over reduced basis states and pick out the correct matrix elements
    #for the Hamiltonian in the reduced basis
    for i, state_i in enumerate(basis_red):
        for j, state_j in enumerate(basis_red):
            H_red[i,j] = H_ori[index_red[i], index_red[j]]

    return H_red


