import numpy as np
from scipy import constants as spc

def gamma_factor(beta):
    return 1/np.sqrt(1-beta**2)

def lambda_of_E(E_kin):
    k_e = spc.m_e*gamma_factor(KeVtoBeta(E_kin))*KeVtoBeta(E_kin)*spc.c/spc.hbar 
    return 1/k_e*2*np.pi

# Materials 
n_materials = {'MoS2': 4.81,
            'Si':3.98,
            'Diamond':2.41,
            'ZrO':2.16,
            'ITO':1.8, 
            #'Polymer': 1.7,
            'BBO': 1.65,
            'Mica':1.6,
            'Silica':1.46}

# Converting Units
def KeVtoBeta(E_KeV,m=spc.m_e):
    E = m*spc.c**2 + E_KeV*1000*spc.e
    return np.sqrt(1-(m**2*spc.c**4)/E**2)

def BetatoKeV(beta,m=spc.m_e):
    E_Joule = m*spc.c**2*np.sqrt(1/(1-beta**2))-m*spc.c**2
    return E_Joule/spc.e/1000

def LambdatoeV(lambda_): 
    return spc.c/lambda_*spc.h/spc.e

def eVtolambda(eV):
    return spc.c/(spc.e*eV/spc.h)

##############
def Cherenkov_angle(beta, n, deg = False):
    if deg:
        return np.rad2deg(np.arccos(1/n/beta))
    else:
        return np.arccos(1/n/beta)

def ElectronDeflectionAngle(beta, theta_cher, wavelength):
    return np.arctan((spc.hbar*2*np.pi/wavelength*np.sin(theta_cher))/(spc.m_e*gamma_factor(beta)*beta*spc.c-spc.hbar*2*np.pi/wavelength*np.cos(theta_cher)))

def Photon_Rate(L, wavelength,Cher_angle):
    return (2*np.pi*spc.alpha*L)/(wavelength**2)*np.sin(Cher_angle)**2

##############
def MFP(probability, thickness):
    return -thickness/np.log(1-probability)

def sum_MFP(ell1,ell2):
    return ell1*ell2/(ell1+ell2)

def sub_MFP(ell,ell1):
    return ell*ell1/(ell1 - ell)

##############
def n_fused_silica(l): #wavelengths in μm please
    return np.sqrt(1 + (0.6961663*l**2)/(l**2-0.0684043**2)+(0.4079426*l**2)/(l**2-0.1162414**2)+(0.8974794*l**2)/(l**2-9.896161**2))

def NA_waveguide(n_core,n_clad):
    return np.sqrt(n_core**2-n_clad**2)
