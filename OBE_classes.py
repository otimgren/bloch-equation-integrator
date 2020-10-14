"""
Classes for use in setting up and solving optical Bloch equations
"""

#Import necessary packages:
import sys
import pickle
import numpy as np
import scipy
from scipy.sparse import csr_matrix

#Custom classes for defining molecular states and some convenience functions for them
sys.path.append('./molecular-state-classes-and-functions/')
from classes import UncoupledBasisState, CoupledBasisState, State
from functions import (make_hamiltonian, make_hamiltonian_B, make_QN, ni_range, vector_to_state,
                        find_state_idx_from_state, make_transform_matrix, matrix_to_states)

#Functions used for integrating OBEs
from OBE_functions import *

class OpticalField:
    """
    Defines an optical field with peak Rabi rate Omega_peak and 
    polarization p_t. The field is taken to be close to resonance for transitions connecting 
    ground_states and excited_states. To define detunings, we use the states ground_main 
    and excited_main if they are provided. Rotating wave approximation and rotating frame
    are assumed.

    inputs:
    Omega_peak = peak Rabi rate [2*pi*Hz]
    p_t = polarization of field as function as function of time (lambda function of t)
    ground_states = list of State objects stating the ground states for the field
    excited_states = list of State objects stating the excited states for the field
    ground_main = ground_state for the main transition being driven (used for calculating Omega
                  and detuning)
    excited_main = ground_state for the main transition being driven (used for calculating Omega
                   and detuning)
    detuning = detuning of field from resonance with the main transition
    Omega_r = Rabi rate profile as function of position so that Omega_r = 1 when at peak Rabi
              rate (lambda function of r where r is a vector in metres)
    E_r = magnitude of oscillating electric field as function of position (lambda function of r)
    """
    #Initialization
    def __init__(self, p_t, ground_states, excited_states,ground_main = None, 
                 excited_main = None, detuning = 0, Omega_r = None, E_r = None):

        self.p_t = p_t
        self.ground_states = ground_states
        self.excited_states = excited_states
        self.ground_main = ground_main
        self.excited_main = excited_main
        self.detuning = detuning
        if Omega_r is None:
            self.Omega_r = lambda t: 2*np.pi*1e6
        else:
            self.Omega_r = Omega_r
        if E_r is None:
            self.E_r = lambda t: 1
        else:
            self.E_r = E_r
        

    #Method for finding the exact eigenstates of a given Hamiltonian that most closely correspond
    #to the excited and ground states defined for the field
    def find_closest_eigenstates(self, H, QN, V_ref):
        """
        inputs:
        H = Hamiltonian whose eigenstates are to be used
        QN =  the basis in which the Hamiltonian is expressed
        V_ref = reference matrix that defines the ordering of the basis states
        """
        ground_states = self.ground_states
        excited_states = self.excited_states
        ground_main = [self.ground_main]
        excited_main = [self.excited_main]

        #Find the exact states corresponding to the given approximate states
        ground_states = find_exact_states(ground_states, H, QN, V_ref = V_ref)
        excited_states = find_exact_states(excited_states, H, QN, V_ref = V_ref)

        #Also find the eigenstates that correspond to the "main" transition
        ground_main = find_exact_states(ground_main, H, QN, V_ref = V_ref)[0]
        excited_main = find_exact_states(excited_main, H, QN, V_ref = V_ref)[0]


        #Redefine the states of the field
        self.ground_states = ground_states
        self.excited_states = excited_states
        self.ground_main = ground_main
        self.excited_main = excited_main

    #Method to generate matrices describing couplings due to the field
    def generate_couplings(self, QN):
        """
        inputs:
        QN = list of quantum numbers that define the basis for the coupling Hamiltonian
        """
        ground_states = self.ground_states
        excited_states = self.excited_states

        #Loop over possible polarizations and generate coupling matrices
        H_list = []
        for i in range(0,3):
            #Generate polarization vector
            pol_vec = np.array([0,0,0])
            pol_vec[i] = 1

            #Generate coupling matrix
            H = optical_coupling_matrix(QN, ground_states, excited_states, pol_vec = pol_vec, 
                                        reduced = False)
            
            #Remove small components
            H[np.abs(H) < 1e-3*np.max(np.abs(H))] = 0

            #Check that matrix is Hermitian
            is_hermitian = np.allclose(H,H.conj().T)
            if not is_hermitian:
                print("Warning: Optical coupling matrix {} is not Hermitian!".format(i))

            #Convert to sparse matrix
            H = csr_matrix(H)

            #Append to list
            H_list.append(H)

        return H_list

    #Method for calculating angular part of matrix element for main transition
    def calculate_ME_main(self):
        ME_main = ED_ME_mixed_state(self.excited_main, self.ground_main, pol_vec = np.array([0,0,1]))
        self.ME_main = ME_main
        return ME_main

    #Method for generating the diagonal matrix that shifts energies in rotating frame
    def generate_D(self, H, QN):
        ground_main = self.ground_main
        excited_main = self.excited_main
        delta = self.detuning

        #Find the transition frequency for main transition
        i_g = QN.index(ground_main)
        i_e = QN.index(excited_main)
        omega0 = np.real(H[i_e,i_e] - H[i_g,i_g])

        #Calculate shift (defining detuning = omega-omega0)
        omega = omega0 + delta

        #Generate the shift matrix
        D = np.zeros(H.shape)
        for excited_state in self.excited_states:
            i = QN.index(excited_state)
            D[i,i] = -omega

        return D

    #Method for finding the time-dependence of Omega based on given Rabi rate profile
    def find_Omega_t(self, r_t):
        Omega_r = self.Omega_r
        Omega_t = lambda t: Omega_r(r_t(t))
        
        return Omega_t
    
    #Method for finding time-dependece of Omega based on given E-field magnitude profile
    def find_Omega_t_from_E_r(self,r_t):
        E_r = self.E_r
        ME_main = np.abs(self.ME_main)
        
        #Calculate effective dipole moment
        Gamma = 1/100e-9 #Natural linewidth in 2*pi*Hz
        f = 3e8/271.7e-9 #Frequency in Hz
        D_eff = (np.sqrt(3*np.pi*8.85e-12*1.05e-34*3e8**3*Gamma/(2*np.pi*f)**3)
                /(1/3e8 * 1e-21)* 0.393430307 * 5.291772e-9/4.135667e-15)  #Hz/(V/cm)

        #Calculate Omega as function of time
        Omega_t = lambda t: (D_eff * E_r(r_t(t)))

        return Omega_t

class MicrowaveField:
    """
    Define a class for microwave fields.

    inputs:
    Omega_peak = peak Rabi rate [2*pi*Hz]
    p_t = polarization of field as function as function of time (lambda function of t)
    ground_states = list of State objects stating the ground states for the field
    excited_states = list of State objects stating the excited states for the field
    ground_main = ground_state for the main transition being driven (used for calculating Omega
                  and detuning)
    excited_main = ground_state for the main transition being driven (used for calculating Omega
                   and detuning)
    detuning = detuning of field from resonance with the main transition
    Omega_r = Rabi rate profile as function of time normalized so that Omega_r = 1 when at peak Rabi
              rate (lambda function of t)
    """
    #Initialization
    def __init__(self, Omega_peak, p_t, Jg, Je, ground_main = None, excited_main = None,
                 Omega_r = None, detuning = 0):
        self.Omega_peak = Omega_peak
        self.p_t = p_t
        self.Jg = Jg
        self.Je = Je

        #By default, the states used as the main transition are taken to be the ones with maximal F
        if ground_main is None:
            self.ground_main = 1*CoupledBasisState(J = Jg, F1 = Jg+1/2,F = Jg+1, mF = 0, I1 = 1/2, I2 = 1/2,
                                                   electronic_state = 'X', P = (-1)**Jg, Omega = 0)
        else:
            self.ground_main = ground_main

        if excited_main is None:
            self.excited_main = 1*CoupledBasisState(J = Je, F1 = Je+1/2,F = Je+1, mF = 0, I1 = 1/2, I2 = 1/2,
                                                   electronic_state = 'X', P = (-1)**Je, Omega = 0)
        else:
            self.excited_main = excited_main
        
        
        self.detuning = detuning
        if Omega_r is None:
            self.Omega_r = lambda t: Omega_peak
        else:
            self.Omega_r = Omega_r

    #Method for finding the exact eigenstates of a given Hamiltonian that most closely correspond
    #to the excited and ground states defined for the field
    def find_closest_eigenstates(self, H, QN, V_ref):
        """
        inputs:
        H = Hamiltonian whose eigenstates are to be used
        QN =  the basis in which the Hamiltonian is expressed
        V_ref = reference matrix that defines the ordering of the basis states
        """
        ground_main = [self.ground_main]
        excited_main = [self.excited_main]

        #Find the eigenstates that correspond to the "main" transition
        ground_main = find_exact_states(ground_main, H, QN, V_ref = V_ref)[0]
        excited_main = find_exact_states(excited_main, H, QN, V_ref = V_ref)[0]

        #Redefine the states of the field
        self.ground_main = ground_main
        self.excited_main = excited_main

    #Method for calculating angular part of matrix element for main transition
    def calculate_ME_main(self):
        ME_main = ED_ME_mixed_state(self.excited_main, self.ground_main, pol_vec = np.array([0,0,1]))
        return ME_main

    #Method to generate matrices describing couplings due to the field
    def generate_couplings(self, QN):
        """
        inputs:
        QN = list of quantum numbers that define the basis for the coupling Hamiltonian
        """
        Jg = self.Jg
        Je = self.Je

        #Loop over possible polarizations and generate coupling matrices
        H_list = []
        for i in range(0,3):
            #Generate polarization vector
            pol_vec = np.array([0,0,0])
            pol_vec[i] = 1

            #Generate coupling matrix
            H = make_H_mu(Jg, Je, QN, pol_vec = pol_vec)
            
            #Remove small components
            H[np.abs(H) < 1e-3*np.max(np.abs(H))] = 0

            #Check that matrix is Hermitian
            is_hermitian = np.allclose(H,H.conj().T)
            if not is_hermitian:
                print("Warning: Microwave coupling matrix {} is not Hermitian!".format(i))

            #Convert to sparse matrix
            H = csr_matrix(H)

            #Append to list of coupling matrices
            H_list.append(H)

        return H_list

    #Method for calculating the transition frequency (in 2pi*Hz)
    def calculate_frequency(self, H, QN):
        ground_main = self.ground_main
        excited_main = self.excited_main
        delta = self.detuning

        #Find the transition frequency for main transition
        i_g = QN.index(ground_main)
        i_e = QN.index(excited_main)
        omega0 = np.real(H[i_e,i_e] - H[i_g,i_g])

        #Calculate shift (defining detuning = omega-omega0)
        omega = omega0 + delta

        return omega

    #Method for generating the diagonal matrix that shifts energies in rotating frame
    def generate_D(self, omega, H, QN, V_ref):
        Je = self.Je

        I_F = 1/2
        I_Tl = 1/2
        Js = [Je]
        excited_states =  [1*CoupledBasisState(F,mF,F1,J,I_F,I_Tl, electronic_state='X', P = (-1)**J, Omega = 0)
                                  for J  in Js
                                  for F1 in ni_range(np.abs(J-I_F),J+I_F+1)
                                  for F in ni_range(np.abs(F1-I_Tl),F1+I_Tl+1)
                                  for mF in ni_range(-F, F+1)
                                 ]

        #Find the exact excited states
        excited_states = find_exact_states(excited_states, H, QN, V_ref = V_ref)

        #Generate the shift matrix
        D = np.zeros(H.shape)
        for excited_state in excited_states:
            i = QN.index(excited_state)
            D[i,i] = -omega

        return D

    #Method for finding the time-dependence of Omega based on given Rabi rate profile
    def find_Omega_t(self, r_t):
        Omega_r = self.Omega_r
        Omega_t = lambda t: Omega_r(r_t(t))
        
        return Omega_t


    

    



                                 





