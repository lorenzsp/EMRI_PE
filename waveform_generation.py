"""
Install environment on Mac:
conda create -n sbi_emri python=3.12
git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
cd FastEMRIWaveforms
conda install -c conda-forge clang_osx-arm64 clangxx_osx-arm64
conda install -c conda-forge liblapacke lapack openblas   
conda install -c conda-forge wget gsl hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests rich matplotlib numba
python scripts/prebuild.py
python setup.py install

git clone https://github.com/mikekatz04/lisa-on-gpu.git
cd lisa-on-gpu        
git reset --hard f042d4f
python setup.py install

git clone git@github.com:gwastro/pycbc.git
cd pycbc
pip install -r requirements.txt
pip install -r companion.txt
pip install .
"""
import numpy as np
from datetime import datetime
import cupy as cp

import time

from few.waveform import GenerateEMRIWaveform, FastSchwarzschildEccentricFlux

from fastlisaresponse import pyResponseTDI, ResponseWrapper
from astropy import units as un

from lisatools.detector import EqualArmlengthOrbits, ESAOrbits

import pycbc.noise
import pycbc.psd
from pycbc.types import FrequencySeries

from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

import pickle
from typing import Tuple

MSUN = 1.98855e30
PC = 3.08567758149137e16
YRSID_SI = 31558149.763545603

TOBs = 3150000 # observation time in seconds
TOB = TOBs/YRSID_SI 
dt = 10
Nsamples = TOBs/dt
frate = 1/dt
deltaf = frate/Nsamples

psd = pycbc.psd.analytical_space.analytical_psd_lisa_tdi_AE_confusion(int(Nsamples), deltaf, 5e-4, tdi=2.0) # check lower flow

eps = 1e-5 # epsilon defined in FEW - contribution of modes to EMRI signal

def get_time_noise():
    noise_A = pycbc.noise.gaussian.noise_from_psd(int(Nsamples),dt,psd)
    noise_E = pycbc.noise.gaussian.noise_from_psd(int(Nsamples),dt,psd)
    return noise_A[2001:], noise_E[2001:], noise_A.sample_times[2001:] # get rid of first samples to match removal of junk in response calculation

def waveform(z):
    use_gpu = True
    # Initialize FEW waveform
    # keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
    inspiral_kwargs={
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    }

    # keyword arguments for inspiral generator (RomanAmplitude)
    amplitude_kwargs = {
        "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
        "use_gpu": use_gpu  # GPU is available in this class
    }

    # keyword arguments for Ylm generator (GetYlms)
    Ylm_kwargs = {
        "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
    }

    # keyword arguments for summation generator (InterpolatedModeSum)
    sum_kwargs = {
        "use_gpu": use_gpu,  # GPU is availabel for this type of summation
        "pad_output": False,
    }

    gen_wave = GenerateEMRIWaveform("FastSchwarzschildEccentricFlux",inspiral_kwargs=inspiral_kwargs,
        amplitude_kwargs=amplitude_kwargs,
        Ylm_kwargs=Ylm_kwargs,
        sum_kwargs=sum_kwargs,
        use_gpu=use_gpu,)

    # define parameters
        
    M = z[0] * 1e5
    mu = z[1] * 10
    p0 = z[2] * 10
    e0 = z[3]
    qK = np.arcsin(z[4]) #this is thetaK but in ecliptic coords (source)
    phiK = z[5] # this is phi (source)
    dist = z[6] * 1e-2
    Phi_phi0 = z[7]
    Phi_r0 = z[8]
    qS = np.arcsin(z[9])
    phiS = z[10]

    a = 0.1  # will be ignored in Schwarzschild waveform
    x0 = 1.0  # will be ignored in Schwarzschild waveform
    Phi_theta0 = 5.0 # will be ignored in Schwarzschild waveform

    ht = gen_wave(
        M,
        mu,
        a,
        p0,
        e0,
        x0,
        dist,
        qS,
        phiS,
        qK,
        phiK,
        Phi_phi0,
        Phi_theta0,
        Phi_r0,
        T=TOB,
        dt=dt,
        eps = eps,
        )

    if len(ht)<Nsamples:
        ht = np.pad(ht, (0, int(Nsamples - len(ht))), 'constant', constant_values=(0))
    if len(ht)>Nsamples:
        ht = ht[:int(Nsamples)]
    t_grid = np.arange(len(ht)) * dt
    return cp.asnumpy(ht), t_grid

def generate_noise():

    noise_A, noise_E, t_grid = get_time_noise()
    t_grid = np.arange(len(noise_A)) * dt
    return np.vstack((t_grid, 1e24 * noise_A, 1e24 * noise_E))

def generate_waveform(z):

    wfm,t_grid = waveform(z)
        
    class GBWave:
        def __init__(self, use_gpu=True):

            if use_gpu:
                self.xp = np
            else:
                self.xp = np

        def __call__(self, T=1.0, dt=10.0):

            return wfm

    gb = GBWave(use_gpu=True)

    T = TOB
    t0 = 10000.0  # time at which signal starts (chops off data at start of waveform where information is not correct)

    # order of the langrangian interpolation
    order = 25

    orbit_file_esa = "EMRI_PE/Github_Repos/lisa-on-gpu/orbit_files/esa-trailing-orbits.h5" # "esa-trailing-orbits.h5"

    orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

    # 1st or 2nd or custom (see docs for custom)
    tdi_gen = "2nd generation"

    index_lambda = 0
    index_beta = 1

    tdi_kwargs_esa = dict(
        orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AET",
    )

    gb_lisa_esa = ResponseWrapper(
        gb,
        T,
        dt,
        index_lambda,
        index_beta,
        t0=t0,
        flip_hx=True,  # set to True if waveform is h+ - ihx
        use_gpu=True,
        remove_sky_coords=True,  # True if the waveform generator does not take sky coordinates
        is_ecliptic_latitude=True,  # False if using polar angle (theta)
        remove_garbage=True,  # removes the beginning of the signal that has bad information
        #orbits=EqualArmlengthOrbits(),
        **tdi_kwargs_esa,
    )

   # Define GB params

    beta = np.arcsin(z[9]) #qS = thetaS but ecliptic
    lam = z[10] #phiS=phiS

    chans = gb_lisa_esa(lam, beta)

    # down-sample and split
    Achan = chans[0]
    Echan = chans[1]
#   Tchan = chans[2]
        
    t_grid = np.arange(len(Achan)) * dt
    return np.vstack((t_grid, 1e24 * cp.asnumpy(Achan), 1e24 * cp.asnumpy(Echan)))

# example injection parameters

M = 5.4
mu = 5.0
p0 = 1.035
e0 = 0.3
sinqK = 0.6471975512
phiK = 2.5471975512
dist = 8.75
Phi_phi0 = 0.0
Phi_r0 = 0.0
sinqS = 0.471975512
phiS = 0.9071975512

ztest = [
    M,
    mu,
    p0,
    e0,
    sinqK,
    phiK,
    dist,
    Phi_phi0,
    Phi_r0,
    sinqS,
    phiS,
]

# Uniform priors
#M = 4,9
#mu = 2,6
#p0 = 0.85,1.4
#e0 = 0.05,0.6
#sinqK = -1,1
#phiK = 0,6.283185307179586
#dist = 1,15
#Phi_phi0 = -3.14159265359,3.14159265359
#Phi_r0 = -3.14159265359,3.14159265359
#sinqS = -1,1
#phiS = 0,6.283185307179586

print(generate_waveform(ztest))
print(generate_noise())

