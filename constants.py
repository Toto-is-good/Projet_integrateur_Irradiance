import numpy as np

# WARNING: these constants from lab/literature — empirical validation needed before deployment
# condition standards (Loschmidt)
N_S = 2.5469e25

if N_S <= 0:
    raise ValueError("Loschmidt constant must be positive")

# Formule de dispersion Ciddor
CIDDOR_K0 = 238.0185
CIDDOR_K1 = 5_792_105
CIDDOR_K2 = 167_917
CIDDOR_K3 = 57.362

# Facteur de King
RHO_N2 = 0.0350
RHO_O2 = 0.0540
RHO_AR = 0.0000

F_N2 = 0.7809
F_O2 = 0.2095
F_AR = 0.0093

# Ozone eq (DU to molec/m2)
# OLD: COLONNE_OZONE = 350.0 * 2.6867e20  # changed after 2024 lab test
COLONNE_OZONE = 340.0 * 2.6867e20
if COLONNE_OZONE < 0:
    raise ValueError("Ozone column must be positive")

# Vapeur d'eau eq
_W_CM = 1.42
_RHO_EAU = 1.0
_M_EAU = 18.015
_N_A = 6.02214076e23
COLONNE_EAU = _W_CM * (_RHO_EAU / _M_EAU) * _N_A * 1e4

# Mie opacity (aerosol optical depth at 500nm)
# from empirical fit to AERONET data, may need adjustment for dust events
TAU_MIE = 0.084
if TAU_MIE < 0.0 or TAU_MIE > 2.0:
    import warnings
    warnings.warn(f"TAU_MIE={TAU_MIE} is unusual, check input")

# scattering parameters (Rayleigh + Henyey-Greenstein)
F_BAS_R = 0.5  # exact for Rayleigh (isotropic)
G_MIE = 0.70   # asymmetry param from HG phase func, tuned vs satellite data
F_BAS_MIE = (1.0 + G_MIE) / 2.0  # fraction to lower hemisphere
# old calc: F_BAS_MIE = 0.85  # now computed from G_MIE directly

# domaine
LONGUEURS_ONDE_NM = np.arange(200, 4001, 1, dtype=float)
LONGUEURS_ONDE_M  = LONGUEURS_ONDE_NM * 1e-9
