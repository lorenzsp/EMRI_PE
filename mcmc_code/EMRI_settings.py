###
# This is the main file that should be edited. 
###
try:
    import cupy as cp
    import numpy as np
    xp = cp
    use_gpu = True
except ImportError:
    import numpy as np
    xp = np
    use_gpu = False

# Intrinsic parameters   
M = 5.4e5;    # Primary mass (units of solar mass)
mu = 50.0;  # Secondary mass (units of solar mass)
a = 0.0;    # Primary spin parameter (a \in [0,1])
p0 = 10.35;   # Initial semi-latus rectum (dimensionless)
e0 = 0.3;   # Initial eccentricity (dimensionless)
iota0 = 0.0;  # Initial inclination angle (with respect to the equatorial plane, (radians))
Y0 = np.cos(iota0);  


dist = 8.75/100


# Angular variables
sinqS = 0.471975512
sinqK = 0.6471975512
qS = np.arcsin(sinqS)
qK = np.arcsin(sinqK)
phiK = 2.5471975512
phiS = 0.9071975512

# Initial angular phases -- positional elements along the orbit. 
Phi_phi0 = 0.0   # Azimuthal phase
Phi_theta0 = 0.0;   # Polar phase
Phi_r0 = 0.0;    # Radial phase

# Waveform params
delta_t = 10.0;  # Sampling interval [seconds]
TOBs = 3150000 # observation time in seconds
T = TOBs/31558149.763545603  # Evolution time [years]

mich = False #mich = True implies output in hI, hII long wavelength approximation

# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)

inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory -- much faster
    "max_init_len": int(1e4),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is available for this type of summation
    "pad_output": True,
}




