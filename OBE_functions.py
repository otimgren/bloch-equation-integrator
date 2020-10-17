"""
Functions for integrating Optical Bloch Equations using the method outlined in
John Barry's thesis, section 3.4
"""

import sys
import numpy as np
sys.path.append('./molecular-state-classes-and-functions/')
from classes import UncoupledBasisState, CoupledBasisState, State
from functions import find_state_idx_from_state
from matrix_element_functions import ED_ME_coupled
from tqdm.notebook import tqdm
from scipy import constants
from numpy import sqrt, exp
# from sympy import exp, sqrt
from scipy.sparse import kron, eye, coo_matrix, csr_matrix
# from numpy import kron, eye
from scipy.special import jv


def generate_sharp_superoperator(M, identity = None):
    """
    Given an operator M in Hilbert space, generates sharp superoperator M_L in Liouville space (see "Optically pumped atoms" by Happer, Jau and Walker)
    sharp = post-multiplies density matrix: |rho@A) = A_sharp @ |rho) 

    inputs:
    M = matrix representation of operator in Hilbert space

    outputs:
    M_L = representation of M in in Liouville space
    """

    if identity == None:
         identity = eye(M.shape[0], format = 'coo')

    M_L = kron(M.T,identity, format = 'csr')

    return M_L

def generate_flat_superoperator(M, identity = None):
    """
    Given an operator M in Hilbert space, generates flat superoperator M_L in Liouville space (see "Optically pumped atoms" by Happer, Jau and Walker)
    flat = pre-multiplies density matrix: |A@rho) = A_flat @ |rho)

    inputs:
    M = matrix representation of operator in Hilbert space

    outputs:
    M_L = representation of M in in Liouville space
    """
    if identity == None:
         identity = eye(M.shape[0], format = 'coo')

    M_L = kron(identity, M, format = 'csr')

    return M_L

def generate_commutator_superoperator(M, identity = None):
    """
    Function that generates the commutator [M,rho] superoperator in Liouville space

    inputs:
    M = matrix representation of operator in Hilbert space that whose commutator with density matrix is being generated

    outputs:
    M_L = representation of commutator with M in in Liouville space
    """

    if identity == None:
         identity = eye(M.shape[0], format = 'coo')

    M_com = generate_flat_superoperator(M, identity=identity)  - generate_sharp_superoperator(M, identity=identity)

    return M_com

def generate_superoperator(A,B):
    """
    Function that generates superoperator representing |A@rho@B) = np.kron(B.T @ A) @ |rho)

    inputs:
    A,B = matrix representations of operators in Hilbert space

    outpus:
    M_L = representation of A@rho@B in Liouville space
    """

    M_L = kron(B.T, A, format = 'csr')

    return M_L

def generate_Ls_from_Hs(H_list):
    """
    Function that takes a list of Hamiltonians and generates the corresponding Lindbladians. 
    Two lists of lindbladians are returned, one based on upper triangular part of the Hamiltonians
    and one based on the lower triangular parts. This helps with adding phase modulation to the 
    simulations.

    inputs:
    H_list = list of Hamiltonians

    outputs:
    Lu_list = Lindbladians generated based on upper triangular parts of H_list
    Ll_list = Lindbladians generated based on lower triangular parts of H_list
    """
    #Generate lists up Lu and Ll by looping over the Hamiltonians
    Lu_list = []
    Ll_list = []

    for H in H_list:
        #Take upper and lower parts of Hamiltonian
        Hu = np.triu(H)
        Hl = np.tril(H)

        #Calculate Lindbladians
        Lu = generate_commutator_superoperator(Hu)
        Ll = generate_commutator_superoperator(Hl)

        #Store Linbladians in lists
        Lu_list.append(Lu)
        Ll_list.append(Ll)

    return Lu_list, Ll_list

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
        ground_state = ground_state.remove_small_components(tol = 1e-5)
        for excited_state in excited_states:
            j = QN.index(excited_state)
            excited_state = excited_state.remove_small_components(tol = 1e-5)

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

def calculate_BR(excited_state, ground_states, tol = 1e-5):
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
        MEs[i] = ED_ME_mixed_state(ground_state.remove_small_components(tol = tol),excited_state.remove_small_components(tol = tol))
    
    #Calculate branching ratios
    BRs = np.abs(MEs)**2/(np.sum(np.abs(MEs)**2))

    return BRs

def collapse_matrices(QN, ground_states, excited_states, gamma = 1, tol = 1e-4):
    """
    Function that generates the collapse matrix for given ground and excited states

    inputs:
    QN = list of states that defines the basis for the calculation
    ground_states = list of ground states that are coupled to the excited states
    excited_states = list of excited states that are coupled to the ground states
    gamma = decay rate of excited states
    tol = couplings smaller than tol/sqrt(gamma) are set to zero to speed up computation

    outputs:
    C_list = list of collapse matrices
    """
    #Initialize list of collapse matrices
    C_list = []

    #Start looping over ground and excited states
    for excited_state in tqdm(excited_states):
        j = QN.index(excited_state)
        BRs = calculate_BR(excited_state, ground_states)
        if np.sum(BRs) > 1:
            print("Warning: Branching ratio sum > 1")
        for ground_state, BR in zip(ground_states, BRs):
            i = QN.index(ground_state)

            if sqrt(BR) > tol:
                #Initialize the coupling matrix
                H = np.zeros((len(QN),len(QN)), dtype = complex)

                H[i,j] = sqrt(BR*gamma)

                C_list.append(H)

    return C_list

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
    rho = np.zeros((len(QN), len(QN)), dtype = complex)

    #Loop over the populated states and set diagonal elements of rho
    for state, pop in zip(states_pop, pops):
        i = QN.index(state)
        rho[i,i] = pop

    return rho

def oscillating_field_magnitude(z,y, z0 = 0, y0 = 0, fwhm_z = 0.0254, fwhm_y = 0.0254, power = 1):
    """
    Function that calculates the magnitude of electric field at position x due to e.g. a 
    microwave horn or laser with a Gaussian intensity profile defined by its width (fwhm) 
    and total power.
    
    inputs:
    z = z position where electric field is to be evaluated (z is along beamline) [m]
    y = y position where electric field is to be evaluated (y perpendicular to beamline and direction 
        of mu-wave propagation) [m]
    z0 = position of the center of the microwave beam [m]
    fwhm_x = full-width-half-maximum of the microwave intensity profile along x [m]
    fwhm_y = full-width-half-maximum of the microwave intensity profile along y [m]
    power = output power in beam [W]
    
    returns:
    E = magnitude of oscillating electric field at x [V/cm]
    """
    
    #Convert FWHM to standard deviation
    sigma_z = fwhm_z/(2*sqrt(2*np.log(2)))
    sigma_y = fwhm_y/(2*sqrt(2*np.log(2)))
    
    #Convert power to amplitude of the Gaussian
    I0 = power/(2*np.pi *sigma_z*sigma_y)

    #Calculate intensity at (0,y,z)
    I_z = I0 * exp(-1/2*((z-z0)/sigma_z)**2) * exp(-1/2*((y-y0)/sigma_y)**2)
    
    #Calculate electric field from intensity (in V/m)
    c = constants.c
    epsilon_0 = constants.epsilon_0
    E = sqrt(2*I_z/(c*epsilon_0))
    
    #Return electric field in V/cm
    return E/100

def multipassed_laser_E_field(z,y, z0 = 0, y0 = 0, fwhm_z = 0.001, fwhm_y = 0.005, power = 50e-3, n_passes = 11, a = 0.001, t = 1):
    """
    Function that calculates the electric field due to a multipassed laser.
    
    inputs:
    z = z position where electric field is to be evaluated (z is along beamline) [m]
    y = y position where electric field is to be evaluated (y perpendicular to beamline and direction 
        of mu-wave propagation) [m]
    z0 = central position of first pass [m]
    y0 = central y-position of all passes [m]
    fwhm_x = full-width-half-maximum of the intensity profile along x [m]
    fwhm_y = full-width-half-maximum of the intensity profile along y [m]
    power = output power in beam [W]
    n_passes = number of passes in multipass
    t = amount of power transmitted in each pass
    
    returns:
    E = magnitude of oscillating electric field at (0,y,z) [V/cm]
    """
    #Initialize electric field value
    E = 0

    #Loop over passes
    for n in range(0,n_passes):
        #Calculate central position of this pass
        z_pass = z0 + n*a

        #Calculate number of reflections the pass at this position has undergone
        if n % 2 == 0:
            n_r = n
        else:
            n_r = n_passes - n

        #Add contribution of this pass to total value of E
        E += oscillating_field_magnitude(z,y,z0 = z_pass,y0 = y0, fwhm_z = fwhm_z, fwhm_y = fwhm_y, power = t**n_r * power)

    return E
    
def calculate_power_needed(Omega, ME, fwhm_z = 0.0254, fwhm_y = 0.0254, D_TlF = 13373921.308037223):
    """
    Function to calculate the microwave power required to get peak Rabi rate Omega
    for a transition with given matrix element when the microwaves have a
    Gaussian spatial profile

    inputs:
    Omega = desired Rabi rate [2*pi*Hz]
    ME = angular part of matrix 
    D_TlF = effective dipole moment for transition (also the non angular part of the ME) [2*pi*Hz/V/cm]
    fwhm_z = full-width-half-max of intensity distribution along z [m]
    fwhm_x = full-width-half-max of intensity distribution along x [m]

    outputs:
    P = power required to reach the given peak Rabi rate for the given beam and transition properties
    """
    
    #Calculate the microwave electric field required (in V/m)
    E =  Omega/(ME*D_TlF) * 100
    
    #Convert E to peak intensity
    c = constants.c
    epsilon_0 = constants.epsilon_0
    I = 1/2 * c * epsilon_0 * E**2
    
    #Convert FWHM to standard deviation
    sigma_z = fwhm_z/(2*sqrt(2*np.log(2)))
    sigma_y = fwhm_y/(2*sqrt(2*np.log(2)))
    
    #Convert power to amplitude of the Gaussian
    P = I * (2*np.pi *sigma_z*sigma_y)
    
    return P


def calculate_decay_rate(excited_state, ground_states):
    """
    Function to calculate decay rates so can check that all excited states have the same decay rates
    """

    #Initialize container fo matrix elements between excited state and ground states
    MEs = np.zeros(len(ground_states), dtype = complex)

    #loop over ground states
    for i, ground_state in enumerate(ground_states):
        MEs[i] = ED_ME_mixed_state(ground_state,excited_state)

    rate = np.sum(np.abs(MEs)**2)
    return rate


def make_H_mu(J1, J2, QN, pol_vec = np.array((0,0,1))):
    """
    Function that generates Hamiltonian for microwave transitions between J1 and J2 (all hyperfine states) for given
    polarization of microwaves. Rotating wave approximation is applied implicitly by only taking the exp(+i*omega*t) part
    of the cos(omgega*t) into account
    
    inputs:
    J1 = J of the lower rotational state being coupled
    J2 = J of upper rotational state being coupled
    QN = Quantum numbers of each index (defines basis fo matrices and vectors)
    pol_vec = vector describing polarization 
    
    returns:
    H_mu = Hamiltonian describing coupling between 
    
    """
    #Figure out how many states there are in the system
    N_states = len(QN) 
    
    #Initialize a Hamiltonian
    H_mu = np.zeros((N_states,N_states), dtype = complex)
    
    
    #Start looping over states and calculate microwave matrix elements between them
    for i in range(0, N_states):
        state1 = QN[i].remove_small_components(tol = 1e-5)
        
        for j in range(i, N_states):
            state2 = QN[j].remove_small_components(tol = 1e-5)
            
            #Check that the states have the correct values of J
            if ((state1.find_largest_component().J == J1 and state2.find_largest_component().J == J2) 
                or (state1.find_largest_component().J == J2 and state2.find_largest_component().J == J1)
                and state1.find_largest_component().electronic_state == state2.find_largest_component().electronic_state):
                #Calculate matrix element between the two states
                H_mu[i,j] = (ED_ME_mixed_state(state1, state2, reduced=False, pol_vec=pol_vec))
                
    #Make H_mu hermitian
    H_mu = (H_mu + np.conj(H_mu.T)) - np.diag(np.diag(H_mu))
    
    
    #return the coupling matrix
    return H_mu


def find_exact_states(states_approx, H, QN, V_ref = None):
    """
    Function for finding the closest eigenstates corresponding to states_approx

    inputs:
    states_approx = list of State objects
    H = Hamiltonian whose eigenstates are used (should be diagonal in basis QN)
    QN = List of State objects that define basis for H
    V_ref = matrix that defines what the matrix representing Qn should look like (used for keeping track of ordering)

    returns:
    states = eigenstates of H that are closest to states_aprox in a list
    """
    states = []
    for state_approx in states_approx:
        i = find_state_idx_from_state(H, state_approx, QN, V_ref = V_ref)
        states.append(QN[i])

    return states

def sidebands(beta,omega_sb = 2*np.pi*1.6e6, n_cut = 15):
    """
    This function provides the modified amplitude for a laser that is being phase modulated by an EOM
    
    inputs:
    beta = modulation depth
    omega_sb = frequency of modulation
    n_cut = order at which the sideband expansion is cut off
    
    outputs:
    sideband_amps(t) = lambda function of time which gives the amplitudes of the sidebands
    """
    
    sideband_amps = lambda t: (jv(0, beta) + np.sum([jv(k,beta)*(np.exp(1j*k*omega_sb*t) + (-1)**k*np.exp(-1j*k*omega_sb*t))
                                                    for k in range(1, n_cut)]))
    
    return sideband_amps

def boltzmann_pop(state, T = 6.2, B = 6.66e9):
    """
    Function that outputs thermal population in given state based on Boltzmann distribution

    inputs:
    T = rotational temperature [K]
    state = state whose population neeeds to be calculated
    B = rotational constant [Hz]

    returns:
    pop = thermal population in state
    """

    #Define some constants
    h = constants.h
    kB = constants.Boltzmann

    #Calculate partition function
    Z = np.sum([4*(2*J+1)*np.exp(-h*B*J*(J+1)/(kB*T)) for J in range(0,100)])

    #Calculate population in given state
    J = state.find_largest_component().J
    pop = np.exp(-h*B*J*(J+1)/(kB*T))/Z

    return pop

def find_boltzmann_pops(states, T = 6.2, B = 6.66e9):
    """
    Function that calculates populations in states based on Boltzmann distribution

    inputs:
    states = list of State objects

    outputs:
    pops = list of populations in each state
    """
    pops = []
    for state in states:
        pops.append(boltzmann_pop(state, T = T, B = B))

    return np.array(pops)


