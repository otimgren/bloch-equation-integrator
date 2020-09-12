"""
Functions for integrating Optical Bloch Equations using the method outlined in
John Barry's thesis, section 3.4
"""

import sys
import numpy as np
sys.path.append('./molecular-state-classes-and-functions/')
from classes import UncoupledBasisState, CoupledBasisState, State
from matrix_element_functions import ED_ME_coupled
from tqdm.notebook import tqdm

def generate_sharp_superoperator(M):
    """
    Given an operator M in Hilbert space, generates sharp superoperator M_L in Liouville space (see "Optically pumped atoms" by Happer, Jau and Walker)
    sharp = post-multiplies density matrix: |rho@A) = A_sharp @ |rho) 

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
    flat = pre-multiplies density matrix: |A@rho) = A_flat @ |rho)

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

def generate_superoperator(A,B):
    """
    Function that generates superoperator representing |A@rho@B) = np.kron(B.T @ A) @ |rho)

    inputs:
    A,B = matrix representations of operators in Hilbert space

    outpus:
    M_L = representation of A@rho@B in Liouville space
    """

    M_L = np.kron(B.T, A)

    return M_L

def generate_rho_vector(rho):
    """
    Function that generates a column vector from a given density matrix, i.e. transforms it to Liouville space

    inputs:
    rho = density matrix

    outputs:
    rho_vec = density matrix in vector form
    """

    rho_vec = rho.flatten()

    return rho_vec

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

def optical_coupling_matrix(QN, ground_states, excited_states, pol_vec = np.array([0,0,1]), reduced = False):
    """
    Function that generates the optical coupling matrix for given ground and excited states

    inputs:
    QN = list of states that defines the basis for the calculation
    ground_states = list of ground states that are coupled to the excited states (i.e. laser is resonant)
    excited_states = list of excited states that are coupled to the ground states

    outputs:
    H = coupling matrix
    """

    #Initialize the coupling matrix
    H = np.zeros((len(QN),len(QN)), dtype = complex)

    #Start looping over ground and excited states
    for ground_state in ground_states:
        i = QN.index(ground_state)
        for excited_state in excited_states:
            j = QN.index(excited_state)

            #Calculate matrix element and add it to the Hamiltonian
            H[i,j] = ED_ME_mixed_state(ground_state, excited_state, pol_vec = pol_vec, reduced = reduced)

    #Make H hermitian
    H = H + H.conj().T

    return H
                
def ED_ME_mixed_state(bra, ket, pol_vec = np.array([1,1,1]), reduced = False):
    """
    Calculates electric dipole matrix elements between mixed states

    inputs:
    bra = state object
    ket = state object
    pol_vec = polarization vector for the light that is driving the transition (the default is useful when calculating branching ratios)

    outputs:
    ME = matrix element between the two states
    """
    ME = 0
    bra = bra.transform_to_omega_basis()
    ket = ket.transform_to_omega_basis()
    for amp_bra, basis_bra in bra.data:
        for amp_ket, basis_ket in ket.data:
            ME += amp_bra.conjugate()*amp_ket*ED_ME_coupled(basis_bra, basis_ket, pol_vec = pol_vec, rme_only = reduced)

    return ME

def calculate_BR(excited_state, ground_states):
    """
    Function that calculates branching ratios from the given excited state to the given ground states

    inputs:
    excited_state = state object representing the excited state that is spontaneously decaying
    ground_states = list of state objects that should span all the states to which the excited state can decay

    returns:
    BRs = list of branching ratios to each of the ground states
    """

    #Initialize container fo matrix elements between excited state and ground states
    MEs = np.zeros(len(ground_states), dtype = complex)

    #loop over ground states
    for i, ground_state in enumerate(ground_states):
        MEs[i] = ED_ME_mixed_state(ground_state,excited_state)
    
    #Calculate branching ratios
    BRs = np.abs(MEs)**2/(np.sum(np.abs(MEs)**2))

    return BRs

def collapse_matrix(QN, ground_states, excited_states):
    """
    Function that generates the collapse matrix for given ground and excited states

    inputs:
    QN = list of states that defines the basis for the calculation
    ground_states = list of ground states that are coupled to the excited states
    excited_states = list of excited states that are coupled to the ground states

    outputs:
    H = collapse matrix
    """

    #Initialize the coupling matrix
    H = np.zeros((len(QN),len(QN)), dtype = complex)

    #Start looping over ground and excited states
    for excited_state in tqdm(excited_states):
        j = QN.index(excited_state)
        BRs = calculate_BR(excited_state, ground_states)
        for ground_state, BR in zip(ground_states, BRs):
            i = QN.index(ground_state)

            H[i,j] = np.sqrt(BR)

    return H

def generate_density_matrix(QN, states_pop, pops):
    """
    Function for generating the density given the list of quantum numbers that define the basis,
    a list of the states that are populated and a list of the populations in the states.

    inputs:
    QN = list of state objects that defines the basis
    states_pop = states that are populated
    pops = populations in the states

    outputs:
    rho = density matrix
    """

    #Initialize the density matrix
    rho = np.zeros((len(QN), len(QN)))

    #Loop over the populated states and set diagonal elements of rho
    for state, pop in zip(states_pop, pops):
        i = QN.index(state)
        rho[i,i] = pop

    return rho






