import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import pvlib

from constants import (
    N_S, CIDDOR_K0, CIDDOR_K1, CIDDOR_K2, CIDDOR_K3,
    RHO_N2, RHO_O2, RHO_AR, F_N2, F_O2, F_AR,
    LONGUEURS_ONDE_M, LONGUEURS_ONDE_NM,
    F_BAS_R, F_BAS_MIE
)

def charger_atmosphere(chemin_fichier):
    # Charge le fichier atm de base
    try:
        df = pd.read_excel(chemin_fichier, sheet_name=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"atmosphere.xlsx not found at {chemin_fichier}")
    except Exception as e:
        raise ValueError(f"Error reading atmosphere.xlsx: {e}")
    
    # expect exactly 901 rows (0-86km @ 50m step)
    h_brut = df.iloc[:901, 0].to_numpy(dtype=float) 
    N_brut = df.iloc[:901, 1].to_numpy(dtype=float)
    
    if len(h_brut) != 901:
        raise ValueError(f"Expected 901 altitude points, got {len(h_brut)}")
    
    # spline 50m -> 1m (may overshoot, will clip later)
    cs = CubicSpline(h_brut, N_brut, bc_type='not-a-knot')
    h_fin = np.arange(0, 86001, 1, dtype=float)
    N_h = np.clip(cs(h_fin), 0.0, None)  # cap negative values from spline overshoot
    
    colonne = float(np.sum(N_h))
    assert colonne > 0, "Atmospheric column density is zero or negative"
    
    return N_h, colonne

def calc_rayleigh(colonne):
    # Ciddor (1996) dispersion + King factor for dry air
    # TODO: could optimize spline here, currently O(n) for every wavelength
    lam = LONGUEURS_ONDE_M
    sigma = 1.0 / (lam * 1e6)
    sigma2 = sigma ** 2
    n_s = 1.0 + (CIDDOR_K1 / (CIDDOR_K0 - sigma2) + CIDDOR_K2 / (CIDDOR_K3 - sigma2)) * 1e-8
    
    def _fk(rho):
        return (6.0 + 3.0 * rho) / (6.0 - 7.0 * rho)
    
    # Air sec (N2, O2, Ar fractions from Bates 1984)
    F_k = float(F_N2 * _fk(RHO_N2) + F_O2 * _fk(RHO_O2) + F_AR * _fk(RHO_AR))

    # section efficace de diffusion Rayleigh
    sigma_R = ((8.0 * np.pi ** 3) * (n_s ** 2 - 1.0) ** 2 / (3.0 * N_S ** 2 * lam ** 4)) * F_k
    # OLD: sigma_R calculation via direct formula was too slow for sweeps
    
    tau_rayleigh = sigma_R * colonne
    assert np.all(tau_rayleigh >= 0), "Rayleigh OD should be non-negative"
    
    return tau_rayleigh

def get_tau_abs(chemin_xlsx, chemin_h2o, ozone_du=350.0, pwv_cm=1.42):
    # Load ozone cross-section (Serdyuchenko et al. 2014)
    try:
        df = pd.read_excel(chemin_xlsx, sheet_name=0)
    except Exception as e:
        raise ValueError(f"Failed to load ozone from {chemin_xlsx}: {e}")
    
    wl_o3 = df.iloc[:, 2].to_numpy(dtype=float)
    xs_o3 = df.iloc[:, 3].to_numpy(dtype=float) * 1e-4  # cm2 -> m2
    
    if len(wl_o3) == 0:
        raise ValueError("Ozone wavelength array is empty")
    
    cs_o3 = CubicSpline(wl_o3, xs_o3, bc_type='not-a-knot', extrapolate=False)
    sigma_o3 = np.zeros(len(LONGUEURS_ONDE_NM))
    
    msk = (LONGUEURS_ONDE_NM >= wl_o3[0]) & (LONGUEURS_ONDE_NM <= wl_o3[-1])
    sigma_o3[msk] = cs_o3(LONGUEURS_ONDE_NM[msk])
    sigma_o3 = np.clip(sigma_o3, 0.0, None)  # spline can overshoot
    
    if ozone_du < 0 or ozone_du > 600:
        import warnings
        warnings.warn(f"Unusual ozone value: {ozone_du} DU")
    
    tau_O3 = sigma_o3 * (ozone_du * 2.6867e20)
    
    # Load water vapor cross-section (HITRAN 2020 via HAPI)
    try:
        sigma_h2o = np.load(chemin_h2o) * 1e-4  # cm2 -> m2
    except FileNotFoundError:
        raise FileNotFoundError(f"H2O cross-section file not found: {chemin_h2o}")
    except Exception as e:
        raise ValueError(f"Error loading H2O data: {e}")
    
    if pwv_cm < 0 or pwv_cm > 10:
        import warnings
        warnings.warn(f"Unusual PWV value: {pwv_cm} cm")
    
    colonne_eau = pwv_cm * (1.0 / 18.015) * 6.02214076e23 * 1e4
    tau_H2O = sigma_h2o * colonne_eau
    
    return tau_O3, tau_H2O

def load_I0():
    # ASTM G173-03 extraterrestrial spectrum (AM0) via pvlib
    # Note: using 'extraterrestrial', NOT 'direct' or 'global' to avoid double-counting atm effects
    try:
        ref = pvlib.spectrum.get_reference_spectra(standard='ASTM G173-03')
    except Exception as e:
        raise RuntimeError(f"Failed to load reference spectrum: {e}")
    
    wl_ref = ref.index.to_numpy(dtype=float)
    I0_ref = ref['extraterrestrial'].to_numpy(dtype=float)
    
    if len(wl_ref) == 0:
        raise ValueError("Reference spectrum is empty")
    
    cs = CubicSpline(wl_ref, I0_ref, bc_type='not-a-knot', extrapolate=False)
    I0 = np.zeros(len(LONGUEURS_ONDE_NM))
    msk = (LONGUEURS_ONDE_NM >= wl_ref[0]) & (LONGUEURS_ONDE_NM <= wl_ref[-1])
    I0[msk] = cs(LONGUEURS_ONDE_NM[msk])
    
    return np.clip(I0, 0.0, None)

def kastenMasseAir(theta_z_deg):
    # Kasten-Young (1989) air mass formula
    # works up to ~96 degrees, fails at grazing incidence
    am = 1.0 / (np.cos(np.deg2rad(theta_z_deg)) + 0.50572 * (96.07995 - theta_z_deg) ** -1.6364)
    return float(am)

def calc_irradiance(I0, tau_R, tau_O3, tau_H2O, zenith_deg, tau_mie=0.1, debug=False):
    # Main irradiance computation: DNI, DHI, GHI
    AVOID_DIVZERO = 1e-30  # epsilon to prevent 0/0 in single-scattering albedo
    
    cos_z = float(np.cos(np.deg2rad(zenith_deg)))
    
    # air mass
    am = kastenMasseAir(zenith_deg)
    if debug: 
        print(f"zenith={zenith_deg:.2f}° | AM={am:.4f}")

    # optical depth profiles
    tau_vert = tau_R + tau_mie + tau_O3 + tau_H2O
    tau_abs_vert = tau_O3 + tau_H2O
    tau_slant = tau_vert * am  # Beer-Lambert path

    # DNI via Beer-Lambert law
    DNI_spec = I0 * np.exp(-tau_slant)
    DNI = float(np.trapezoid(DNI_spec, LONGUEURS_ONDE_NM))

    # DHI via single-scattering approximation
    # old approach: DHI = I0 * (1 - exp(-tau))  [missing cos_z and abs attenuation]
    omega_R = tau_R / (tau_vert + AVOID_DIVZERO)  # SSA for Rayleigh
    omega_Mie = tau_mie / (tau_vert + AVOID_DIVZERO)  # SSA for Mie
    f_down = omega_R * F_BAS_R + omega_Mie * F_BAS_MIE  # fraction to lower hemisphere

    DHI_spec = I0 * cos_z * (1.0 - np.exp(-tau_slant)) * f_down * np.exp(-tau_abs_vert)
    DHI = float(np.trapezoid(DHI_spec, LONGUEURS_ONDE_NM))

    # GHI = DNI×cos(z) + DHI (exact decomposition)
    GHI_spec = DNI_spec * cos_z + DHI_spec
    GHI = float(np.trapezoid(GHI_spec, LONGUEURS_ONDE_NM))

    return DNI_spec, DHI_spec, GHI_spec, DNI, DHI, GHI
