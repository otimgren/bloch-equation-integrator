#This file contains a function used for solving OBEs for CeNTREX

#Import necessary packages:
import sys
import pickle
import numpy as np
from numpy import triu, tril
import scipy
from scipy.sparse import csr_matrix, coo_matrix#, triu, tril
from scipy.sparse.linalg import onenormest
import timeit
import pickle


#Custom classes for defining molecular states and some convenience functions for them
sys.path.append('./molecular-state-classes-and-functions/')
from classes import UncoupledBasisState, CoupledBasisState, State
from functions import (make_hamiltonian, make_hamiltonian_B, make_QN, ni_range, vector_to_state,
                        find_state_idx_from_state, make_transform_matrix, matrix_to_states, reorder_evecs)

#Functions used for integrating OBEs
from OBE_functions import *

#Custom classes
from OBE_classes import OpticalField, MicrowaveField

#Function for matrix exponentiation
from expokitpy import py_zgexpv

from functools import partial

def OBE_integrator(r0 = np.array((0.,0.,0.)),  r1 = np.array((0,0,3e-2)), v = np.array((0,0,200)),  #Parameters for defining position of molecule as function of time:
                   X_states = None, B_states = None, #States that are needed for simulation
                   microwave_fields = None, laser_fields = None,
                   Gamma = 2*np.pi*1.6e6,
                   states_pop = None, pops = None,
                   Nsteps = int(5e3),
                   verbose = True,
                  ):
    """
    Function that integrates optical Bloch equations for TlF in Centrex.

    The structure of the code is as follows:
    0. Define position of molecule as function of time. Makes it easier to
       convert spatial dependence of e.g. laser/microwave intensity to a
       time dependence.

    1. Define the internal Hamiltonian of the molecule. The Hamiltonian
       for the X- and B-states of TlF is fetched from file and diagonalized.
       The eigenstates of the Hamiltonians are used as the basis in the rest
       of the calculation. The size of the basis is reduced by keeping only 
       states that couple to one of the time-dependent fields, or which can
       be reached via spontaneous decay from the excited state.

        1.1 Define X-state Hamiltonian and find the diagonal basis
        1.2 Define B-state Hamiltonian and find the diagonal basis

    2. Microwave couplings. Construct Hamiltonian for microwave couplings between different 
       rotational states within X.

    3. Optical couplings. Construct the Hamiltonian for laser couplings between X_states_laser and
       B_states_laser.
    
    4. Generate the total Hamiltonian that contains the couplings due to the fields, and the
       internal molecular Hamiltonian

    5. Generate collapse operators that describe spontaneous decay

    6. Generate initial density matrix

    7. Transform to Liouville space

    8. Time-evolution

    inputs:
    r0 = initial position of molecule [m]
    r1 = final positon of molecule [m]
    v = velocity of molecule [m/s]
    X_states = list of states in X-state of TlF used in the simulation (list of State objects)
    B_states = list of states in B-state of TlF used in the simulation (list of State objects)
    laser_fields = list of OpticalField objects (see OBE_classes.py)
    microwave_fields = list of MicrowaveField objects (see OBE_classes.py)
    states_pops = states that are initially populated
    pops = initial populations of states_pops. If none, Boltzmann distribution assumed

    outputs:
    t_array = array of times at which we have datapoints
    pop_results = populations in each state at the times stored in t_array

    """
    ### 0. Define molecular position as function of time ###
    def molecule_position(t, r0, v):
        """
        Function that returns position of molecule at a given time for given initial position and 
        velocity.
        
        inputs:
        t = time in seconds
        r0 = position of molecule at t = 0 in meters
        v = velocity of molecule in meters per second
        
        returns:
        r = position of molecule in metres
        """
        r =  r0 + v*t
        
        return r

    #Define a lambda function that gives position as function of time
    r_t = lambda t: molecule_position(t, r0, v)

    #Calculate total time for simulation [s]
    z0 = r0[2]
    z1 = r1[2]
    vz = v[2]
    T = np.abs((z1-z0)/vz)

    # If want printed output, print it
    if verbose:
        print("Total simulation time is {:.3E} s".format(T))
    
    ### 1. Internal Hamiltonian
    ## 1.1 X-state of TlF

    #Load Hamiltonian from file
    H_X_uc = make_hamiltonian("./utilities/TlF_X_state_hamiltonian_J0to4.pickle")

    #Hamiltonian on file is in uncoupled angular momentum basis. Transform it to coupled.
    #Load transform matrix
    with open("./utilities/UC_to_C_j0to4.pickle","rb") as f:
        S_trans = pickle.load(f)

    #Transform matrix
    E = np.array((0,0,0))
    B = np.array((0,0,0.001)) #Very small magnetic field is used to ensure mF is a good quantum number 
    H_X =  S_trans.conj().T @ H_X_uc(E, B) @ S_trans

    #Define basis to which the Hamiltonian was transformed
    Jmin = 0
    Jmax = 4
    I_F = 1/2
    I_Tl = 1/2
    QN_X = [CoupledBasisState(F,mF,F1,J,I_F,I_Tl, electronic_state='X', P = (-1)**J, Omega = 0)
        for J  in ni_range(Jmin, Jmax+1)
        for F1 in ni_range(np.abs(J-I_F),J+I_F+1)
        for F in ni_range(np.abs(F1-I_Tl),F1+I_Tl+1)
        for mF in ni_range(-F, F+1)
        ]

    #Diagonalize the Hamiltonian (also making sure V is as close as identity matrix as possible
    # in terms of ordering of states)
    D, V = np.linalg.eigh(H_X)
    V_ref_X = np.eye(V.shape[0])
    D, V = reorder_evecs(V,D,V_ref_X)
    H_X_diag = V.conj().T @ H_X @ V

    #Define new basis based on eigenstates of H_X:
    QN_X_diag = matrix_to_states(V, QN_X)

    #Sometimes only a subset of states is needed for the simulation. Determine the X-states
    #that are needed here.
    if X_states is None:
        ground_states = QN_X_diag
    else:
        ground_states = find_exact_states(X_states, H_X_diag, QN_X_diag, V_ref = V_ref_X)

    #Find the Hamiltonian in the reduced basis
    H_X_red = reduced_basis_hamiltonian(QN_X_diag, H_X_diag, ground_states)

    #Set small off diagonal terms to zero
    H_X_red[np.abs(H_X_red) < 0.1] = 0

    ## 1.2 B-state of TlF

    #Load Hamiltonian from file
    H_B = make_hamiltonian_B("./utilities/B_hamiltonians_symbolic_coupled_P_1to3.pickle")

    #Define the basis that the Hamiltonian is in
    Jmin = 1
    Jmax = 3
    I_F = 1/2
    I_Tl = 1/2
    Ps = [-1, 1]
    QN_B = [CoupledBasisState(F,mF,F1,J,I_F,I_Tl,P = P, Omega = 1, electronic_state='B')
        for J  in ni_range(Jmin, Jmax+1)
        for F1 in ni_range(np.abs(J-I_F),J+I_F+1)
        for F in ni_range(np.abs(F1-I_Tl),F1+I_Tl+1)
        for mF in ni_range(-F, F+1)
        for P in Ps
        ]

    #Diagonalize the Hamiltonian
    D,V = np.linalg.eigh(H_B)
    V_ref_B = np.eye(H_B.shape[0])
    D, V = reorder_evecs(V,D,V_ref_B)
    H_B_diag = V.conj().T @ H_B @ V

    #Define new basis based on eigenstates of H_B
    QN_B_diag = matrix_to_states(V, QN_B)

    #Sometimes only a subset of states is needed for the simulation. Determine the X-states
    #that are needed here.
    if B_states is None:
        excited_states = QN_B_diag
    else:
        excited_states = find_exact_states(B_states, H_B_diag, QN_B_diag, V_ref=V_ref_B)

    #Find the Hamiltonian in the reduced basis
    H_B_red = reduced_basis_hamiltonian(QN_B_diag, H_B_diag, excited_states)

    #Set small off diagonal terms to zero
    H_B_red[np.abs(H_B_red) < 0.1] = 0

    ## 1.3 Define total internal Hamiltonian
    H_int = scipy.linalg.block_diag(H_X_red, H_B_red)
    V_ref_int = np.eye(H_int.shape[0])

    #Define Hamiltonian in the rotating frame (transformation not applied yet)
    H_rot = H_int

    #Define QN for the total Hamiltonian that includes both X and B
    QN = ground_states + excited_states

    if verbose:
        print("Diagonal of H_int:")
        print(np.diag(H_int)/(2*np.pi))

    ### 2. Couplings due to microwaves
    #If there are microwave fields, loop over them. Otherwise skip this section.
    if microwave_fields is not None:
        microwave_couplings = []
        D_mu = np.zeros(H_int.shape)
        omegas = []
        for microwave_field in microwave_fields:
            #Find the exact ground and excited states for the field
            microwave_field.find_closest_eigenstates(H_rot, QN, V_ref_int)

            #Find the coupling matrices due to the laser
            H_list = microwave_field.generate_couplings(QN)

            #Find some necessary parameters and then define the coupling matrix as a function of time
            Omega = microwave_field.Omega_peak #Rabi rate
            Omega_t = microwave_field.find_Omega_t(r_t) #Time dependence of Rabi rate
            p_t = microwave_field.p_t #Time dependence of polarization of field
            ME_main = microwave_field.calculate_ME_main() #Angular part of ME for main transition
            omega =microwave_field.calculate_frequency(H_rot, QN) #Calculate frequency of transition
            D_mu =microwave_field.generate_D(omega,H_rot, QN, V_ref_int) #Matrix that shifts energies for rotating frame
            H_rot += D_mu
    

            #Define the coupling matrix as function of time
            def H_mu_t_func(H_list, Omega, ME_main, p_t, Omega_t, t):
                return (Omega_t(t)/ME_main
                        *(triu(H_list[0]*p_t(t)[0] + H_list[1]*p_t(t)[1] + H_list[2]*p_t(t)[2])
                          + tril(H_list[0]*p_t(t)[0].conj() + H_list[1]*p_t(t)[1].conj() + H_list[2]*p_t(t)[2].conj())))

            H_mu_t = partial(H_mu_t_func, H_list, Omega, ME_main, p_t, Omega_t)
            
            # print(H_mu_t(0))

            microwave_couplings.append(H_mu_t)

            #Print output for checks
            if verbose:
                print("ME_main = {:.3E}".format(ME_main))
                i_e = QN.index(microwave_field.excited_main)
                i_g = QN.index(microwave_field.ground_main)
                print(H_mu_t(T/2)[i_e,i_g]/(2*np.pi*1e6))
                print(Omega_t(T/2))

        #Generate function that gives couplings due to all microwaves
        def H_mu_tot_t(t):
            H_mu_tot = np.zeros(H_rot.shape)
            for H_mu_t in microwave_couplings:
                H_mu_tot = H_mu_tot + H_mu_t(t)
            return H_mu_tot

        if verbose:
            with open("H_mu_tot.pickle",'wb+') as f:
                pickle.dump(H_mu_tot_t(T/2),f)

        #Shift energies in H_int in accordance with the rotating frame
        #H_rot = H_rot + D_mu

    else:
        H_mu_tot_t = lambda t: np.zeros(H_rot.shape)

    if verbose:
        time = timeit.timeit("H_mu_tot_t(T/2)", number = 10, globals = locals())/10
        print("Time to generate H_mu_tot_t: {:.3e} s".format(time))
        H_test = H_mu_tot_t(T/2)
        non_zero = H_test[np.abs(H_test) > 0].shape[0]
        print("Non-zero elements at T/2: {}".format(non_zero))

    ### 3. Optical couplings due to laser
    #If there are laser fields, loop over them. Otherwise skip this section
    if laser_fields is not None:
        optical_couplings = []
        D_laser = np.zeros(H_rot.shape)
        for laser_field in laser_fields:
            #Find the exact ground and excited states for the field
            laser_field.find_closest_eigenstates(H_rot, QN, V_ref_int)

            #Find the coupling matrices due to the laser
            H_list = laser_field.generate_couplings(QN)

            #Find some necessary parameters and then define the coupling matrix as a function of time
            p_t = laser_field.p_t #Time dependence of polarization of field (includes phase modulation)
            ME_main = laser_field.calculate_ME_main() #Angular part of ME for main transition
            Omega_t = laser_field.find_Omega_t(r_t) #Time dependence of Rabi rate
            D_laser +=laser_field.generate_D(H_rot, QN) #Matrix that shifts energies for rotating frame
            H_rot = H_rot + laser_field.generate_D(H_rot, QN)


            #Define the optical couplings as function of time
            def H_oc_t_func(H_list, ME_main, p_t, Omega_t, t):
                 return (Omega_t(t)/ME_main
                        *(triu(H_list[0]*p_t(t)[0] + H_list[1]*p_t(t)[1] + H_list[2]*p_t(t)[2])
                        + tril(H_list[0]*p_t(t)[0].conj() + H_list[1]*p_t(t)[1].conj() + H_list[2]*p_t(t)[2].conj()))
                        )

            H_oc_t = partial(H_oc_t_func, H_list, ME_main, p_t, Omega_t)

            optical_couplings.append(H_oc_t)

            #Print output for checks
            if verbose:
                print("ME_main = {:.3E}".format(ME_main))
                i_e = QN.index(laser_field.excited_main)
                i_g = QN.index(laser_field.ground_main)
                print(H_oc_t(T/2)[i_e,i_g]/(2*np.pi*1e6))
                print(Omega_t(T/2)/(1e6*2*np.pi))


            #Generate lambda function due to all lasers
            #H_oc_tot_t = lambda t: np.sum([H_oc_t(t) for H_oc_t in optical_couplings])

        def H_oc_tot_t(t):
            H_oc_tot = np.zeros(H_rot.shape)
            for H_oc_t in optical_couplings:
                H_oc_tot = H_oc_tot + H_oc_t(t)
            return H_oc_tot

        #Shift energies in H_rot in accordance with the rotating frame
        #Also shift the energies so that ground_main is at zero energy
        i_g = QN.index(laser_field.ground_main)
        H_rot = H_rot  - np.eye(H_rot.shape[0])*H_rot[i_g,i_g] # + D_laser

    #If no laser fields are defined, set coupling matrix to zeros
    else:
        H_oc_tot_t = lambda t: np.zeros(H_rot.shape)

    if verbose:
        time = timeit.timeit("H_oc_tot_t(T/2)", number = 10, globals = locals())/10
        print("Time to generate H_oc_tot_t: {:.3e} s".format(time))
        print("Diagonal of H_rot in rotating frame of laser:")
        print(np.diag(H_rot)/(2*np.pi))
        print("D_laser:")
        print(np.diag(D_laser))
        with open("H_oc_tot.pickle",'wb+') as f:
            pickle.dump(H_oc_tot_t(T/2.3156165),f)        

    ### 4. Total Hamiltonian
    #Define the total Hamiltonian (including the oscillating fields) in the rotating frame
    # as a function of time
    H_tot_t = lambda t: coo_matrix(H_rot + H_oc_tot_t(t) + H_mu_tot_t(t))

    if verbose:
        time = timeit.timeit("H_tot_t(T/2)", number = 10, globals = locals())/10
        print("Time to generate H_tot_t: {:.3e} s".format(time))
        print("Diagonal of H_tot_t(T/2) in rotating frame of laser:")
        print(np.diag(H_tot_t(T/2).toarray())/(2*np.pi))
        # print("D_laser:")
        # print(np.diag(D_laser))

    ### 5. Collapse operators
    #Here we generate the matrices that describe spontaneous decay from the excited
    #states to the ground states
    C_list = collapse_matrices(QN, ground_states, excited_states, gamma = Gamma)

    ### 6. Initial populations
    #Find the exact forms of the states that are initially populated
    states_pop = find_exact_states(states_pop, H_rot, QN, V_ref = V_ref_int)

    #If populations in states are not specified, assume Boltzmann distribution
    if pops is None:
        pops = find_boltzmann_pops(states_pop)
        pops = pops/np.sum(pops)
    
    #Generate initial density matrix
    rho_ini = generate_density_matrix(QN,states_pop,pops)

    if verbose:
        print("Initial population in")
        states_pop[1].print_state()
        print("is {:.5f}".format(pops[1]))

    ### 7. Transfer to Liouville space
    #We transfer to Liouville space where the density matrix is a vector
    #and time-evolution is found by matrix exponentiation

    #Generate the Lindbladian
    #First compute the part that contains the spontaneous decay
    L_collapse = np.zeros((len(QN)**2,len(QN)**2), dtype = complex)
    for C in tqdm(C_list):
        L_collapse += (generate_superoperator(C,C.conj().T)
                        -1/2 * (generate_flat_superoperator(C.conj().T @ C) + 
                                generate_sharp_superoperator(C.conj().T @ C)))
    #Make the collapse operator into a sparse matrix
    L_collapse = csr_matrix(L_collapse)

    #Define a function that gives the time dependent part of the Liouvillian
    #at time t
    L_t = lambda t: (-1j*generate_commutator_superoperator(H_tot_t(t)) 
                        + L_collapse)

    if verbose:
        time = timeit.timeit("L_t(T/2)", number = 10, globals = locals())/10
        print("Time to generate L_t: {:.3e} s".format(time))

    ### 8. Time-evolution
    #Here we perform the time-evolution of the system

    #Set rho vector to its initial value
    rho_vec = generate_rho_vector(rho_ini)

    #Set number of steps and calculate timestep
    dt = T/Nsteps

    #Generate array of times
    t_array = np.linspace(0,T,Nsteps)

    #Pre-calculate some parameters for the matrix exponentiation
    #Calculate onenorm estimate
    norm = onenormest(L_t(T/2))

    #Calculate wsp and iwsp
    m = 20 #maximum size of Krylov subspace
    n = rho_vec.shape[0]
    wsp = np.zeros(7+n*(m+2)+5*(m+2)*(m+2),dtype=np.complex128)
    iwsp = np.zeros(m+2, dtype=np.int32)

    #Array for storing results
    pop_results = np.zeros((len(QN), len(t_array)), dtype = float)
    pop_results[:,0] = np.real(np.diag(rho_ini))

    #Loop over timesteps
    for i, t_i in enumerate(tqdm(t_array[1:])):
        #Calculate the Lindbladian at this time
        L_sparse = L_t(t_i)

        #Time evolve the density vector
        rho_vec = py_zgexpv(rho_vec, L_sparse, t = dt, anorm = norm, wsp = wsp, iwsp = iwsp,
                            m = m)

        #Convert back to density matrix
        rho = rho_vec.reshape(len(QN),len(QN))

        #Find populations in each state
        pop_results[:,i+1] = np.real(np.diag(rho))

    if verbose:
        time = timeit.timeit("py_zgexpv(rho_vec, L_sparse, t = dt, anorm = norm, wsp = wsp, iwsp = iwsp,m = m)", number = 10, globals = locals())/10
        print("Time for exponentiating: {:.3E}".format(time))

    return t_array, pop_results










           






