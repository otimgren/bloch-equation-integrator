"""
Functions for integrating Optical Bloch Equations using the method outlined in
John Barry's thesis, section 3.4
"""

#Import packages
import numpy as np

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


