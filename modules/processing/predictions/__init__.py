import numpy as np
import os
from scipy import constants as spc
from scipy import interpolate
from scipy.special import jv 
from scipy.optimize import root
from processing.predictions.Uf_basics import *

def sim_importer(key, y_size, correct_alpha = 1,E_kin = 200, to_SI = True):
    urad,eV,P = np.loadtxt(os.path.join(os.path.dirname(__file__),'EELS_sim_eV_over_urad/' + key),unpack = True)
    
    # Look at editor to see when it starts repeating
    # 791 lines!
    #y_size  = 791
    x_size = urad.size/y_size
    print('x_size = {}'.format(x_size))
    urad = urad.reshape(-1,y_size)
    print('Delta urad = {} urad * {} = {} rad'.format(urad[1,0]- urad[0,0],correct_alpha,(urad[1,0]- urad[0,0])*correct_alpha))
    urad *= correct_alpha 
    eV = eV.reshape(-1,y_size)
    P = P.reshape(-1,y_size)
    delta_urad = urad[1,0]-urad[0,0] 
    delta_eV = eV[0,1] - eV[0,0]
    print('Delta eV = {} eV'.format(delta_eV))
    if to_SI:
        print('Converting dP/dkdE a.u to dP per rad & eV')
        print('Assuming {} keV electrons'.format(E_kin))
        P *= spc.physical_constants['Bohr radius'][0] # per (1/a_0) to per (1/m)
        P /= (spc.hbar**2/(spc.m_e*(spc.physical_constants['Bohr radius'][0]**2))/spc.e) # E = 1 a.u = 27eV
        P *= 1e-6*2*np.pi/lambda_of_E(E_kin) # k vector corresponding to 1 urad kick, given by k-vector of electron 
    return P, urad, eV

def interpol(arr, key1, key2):
    return lambda x: interpolate.splev(x,interpolate.splrep(arr[key1], arr[key2], s=0), der=0)

props = {'Si':{'n':'epsillon_data/Aspnes_n_appended.csv',
              'k':'epsillon_data/Aspnes_k_appended.csv'},
         'SiO2':{'n':'epsillon_data/Rodriguez-de Marcos_n.csv',
              'k':'epsillon_data/Rodriguez-de Marcos_k.csv'}}

Andrea_sims = {'P(Q,om)_Si.dat':791, # 791 energy bins
               'P(Q,om)_Si_convolved.dat':901,
               'P(Q,om)_Si_convolved_experimental_ZLP_Schlitz 2mm_disp. 0.1 eVpCh.dat':901,
               'P(Q,om)_Si_convolved_experimental_ZLP_Schlitz 1mm_disp. 0.1 eVpCh_extraktor3000V.dat':901}

for material in props: 
    for key in props[material]:
        props[material][key] = np.genfromtxt(os.path.join(os.path.dirname(__file__),props[material][key]),delimiter = ',',names=True)
        props[material][key]['wl'] *= 1e-6
        
n = {'Si':interpol(props['Si']['n'],'wl','n'),
    'SiO2':interpol(props['SiO2']['n'],'wl','n')}

k = {'Si':interpol(props['Si']['k'],'wl','k'),
    'SiO2':interpol(props['SiO2']['k'],'wl','k')}
        
'''
Si_n  = np.genfromtxt('Aspnes_n_appended.csv',delimiter = ',',names=True)
Si_k  = np.genfromtxt('Aspnes_k_appended.csv',delimiter = ',',names=True)
SiO2_n  = np.genfromtxt('Rodriguez-de Marcos_n.csv',delimiter = ',',names=True)
SiO2_k  = np.genfromtxt('Rodriguez-de Marcos_k.csv',delimiter = ',',names=True)


# Getting an interpolation of the refractive index and it's derivative 
n_si  = lambda x: interpolate.splev(x,interpolate.splrep(1e-6*Si_n['wl'], Si_n['n'], s=0), der=0)
n_sip = lambda x: interpolate.splev(x,interpolate.splrep(1e-6*Si_n['wl'], Si_n['n'], s=0), der=1)
k_si  = lambda x: interpolate.splev(x,interpolate.splrep(1e-6*Si_k['wl'], Si_k['k'], s=0), der=0)
k_sip = lambda x: interpolate.splev(x,interpolate.splrep(1e-6*Si_k['wl'], Si_k['k'], s=0), der=1)
'''