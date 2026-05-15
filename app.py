import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pvlib
from pvlib.spectrum import spectrl2
from pvlib.atmosphere import get_relative_airmass
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline
from timezonefinder import TimezoneFinder
import pytz
import os
import io

from backend import charger_atmosphere, calc_rayleigh, get_tau_abs, load_I0, calc_irradiance
from constants import LONGUEURS_ONDE_NM

st.set_page_config(page_title="Modélisateur d'irradiance solaire", layout="wide")

# --- PWV HELPER ---
def calc_pwv(temp_c, rh_pct):
    # Gueymard 1994 estimation (returns precipitable water in cm)
    return float(pvlib.atmosphere.gueymard94_pw(temp_c, rh_pct))

# --- SPECTRL2 SPECTRAL ANALYSIS HELPER ---
#def compute_spectrl2_spectrum(zenith_angle_deg, aod500, ozone_atm_cm, pwv_cm, pressure_pa=101300):
    """
    Compute spectral GHI using pvlib's SPECTRL2 (Bird Smith) model with fixed defaults.
    Interpolates from 122 wavelengths to 3801 points for smooth rendering.
    
    Fixed SPECTRL2 parameters:
      - Surface tilt: 37.0°
      - Angle of incidence: 23.50°
      - Ground albedo: 0.2
      - Surface pressure: 101300 Pa
      - Day of year: 1 (January 1st, fixed for consistency)
    
    Configurable parameters:
      - zenith_angle_deg: Solar zenith angle
      - aod500: Aerosol optical depth at 500 nm
      - ozone_atm_cm: Ozone in atm-cm (0.34 = 340 DU)
      - pwv_cm: Precipitable water vapor in cm
    
    Parameters
    ----------
    zenith_angle_deg : float
        Solar zenith angle in degrees
    aod500 : float
        Aerosol optical depth at 500 nm
    ozone_atm_cm : float
        Ozone column in atm-cm (0.34 = 340 DU)
    pwv_cm : float
        Precipitable water vapor in cm
    pressure_pa : float
        Atmospheric pressure in Pa (default 101300)
    
    Returns
    -------
    wavelengths_interp : np.ndarray (3801,)
        Interpolated wavelengths in nm
    ghi_interp : np.ndarray (3801,)
        Interpolated Global Horizontal Irradiance in W/m²/nm at each wavelength
    """
    # Calculate relative airmass using Kasten-Young model
    relative_airmass = get_relative_airmass(zenith_angle_deg, model='kasten1966')
    
    # Fixed SPECTRL2 parameters per user specification
    surface_tilt_deg = 0.0
    aoi_fixed = zenith_angle_deg
    ground_albedo = 0.2
    dayofyear = 1  # January 1st
    
    # Call SPECTRL2 model with fixed defaults
    result = spectrl2(
        apparent_zenith=zenith_angle_deg,
        aoi=aoi_fixed,
        surface_tilt=surface_tilt_deg,
        ground_albedo=ground_albedo,
        surface_pressure=pressure_pa,
        relative_airmass=relative_airmass,
        precipitable_water=pwv_cm,
        ozone=ozone_atm_cm,
        aerosol_turbidity_500nm=aod500,
        dayofyear=dayofyear
    )
    
    # Extract wavelengths and Global Tilted Irradiance (poa_global)
    wavelengths_s2 = np.asarray(result['wavelength']).flatten()  # nm, 122 points
    ghi_s2 = np.asarray(result['poa_global']).flatten()  # W/m²/nm, 122 points
    
    # Interpolate to 3801 points for smooth rendering
    cs = CubicSpline(wavelengths_s2, ghi_s2, bc_type='not-a-knot')
    wavelengths_interp = np.arange(300, 4001, 1, dtype=float)  # 300-4000 nm, 1 nm steps
    ghi_interp = cs(wavelengths_interp)
    ghi_interp = np.clip(ghi_interp, 0.0, None)  # Clip to physical range
    
    return wavelengths_interp, ghi_interp

# --- INITIALIZATION & CACHING ---
MIN_ELEVATION_THRESHOLD = 0.01  # sun below this angle = night
EPSILON_WAVELENGTH = 1e-30      # avoid 0/0 in calculations
MAX_SWEEP_POINTS = 9000          # prevent excessive computation

@st.cache_resource(show_spinner="Chargement des données atmosphériques...")
def initialiser_systeme():
    # Load base constants/cross sections that don't depend on PWV/Ozone
    # Error handling: ensure files are accessible before expensive operations
    CHEMIN_XLSX = 'data/atmosphere.xlsx'
    
    try:
        if not os.path.exists(CHEMIN_XLSX):
            st.error(f"File not found: {CHEMIN_XLSX} — check data/ directory")
            return None, None, None, None
        
        N_h, colonne = charger_atmosphere(CHEMIN_XLSX)
        tau_R = calc_rayleigh(colonne)
        I0 = load_I0()
        
        # sanity checks
        assert len(tau_R) == 3801, f"Expected 3801 wavelengths, got {len(tau_R)}"
        assert np.all(I0 >= 0), "I0 should be non-negative"
        
        return N_h, colonne, tau_R, I0
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None, None, None, None

# Module-level constants for error handling and edge cases

@st.cache_data(show_spinner=False)
def obtenir_epaisseurs_optiques_absorption(ozone_du, pwv_cm):
    CHEMIN_XLSX = 'data/atmosphere.xlsx'
    CHEMIN_H2O = 'data/h2o_xsec_1nm.npy'
    
    try:
        if not os.path.exists(CHEMIN_XLSX) or not os.path.exists(CHEMIN_H2O):
            st.warning(f"Missing absorption data files")
            return None, None
        
        # validation
        if ozone_du < 0 or ozone_du > 600:
            st.warning(f"Unusual ozone value: {ozone_du} DU")
        if pwv_cm < 0 or pwv_cm > 10:
            st.warning(f"Unusual PWV value: {pwv_cm} cm")
        
        tau_O3, tau_H2O = get_tau_abs(CHEMIN_XLSX, CHEMIN_H2O, ozone_du=ozone_du, pwv_cm=pwv_cm)
        return tau_O3, tau_H2O
    except Exception as e:
        st.error(f"Error loading absorption data: {e}")
        return None, None

# Initialize Base Data
try:
    system_data = initialiser_systeme()
    if system_data[0] is not None:
        N_h, colonne_tot, tau_R, I0 = system_data
    else:
        st.stop()
except Exception as e:
    st.error(f"Fatal error during initialization: {e}")
    st.stop()


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choisir le mode", [
    "1. Profil unique", 
    "2. Analyse avancée (comparaison / journée réelle)",
    "3. Analyse spectrale"
])

def render_configuration_ui(show_spectral_elevation_controls=False, show_real_day_controls=False):
    st.subheader("Paramètres atmosphériques")
    
    # Initialize session state for configurations
    if 'configs' not in st.session_state:
        st.session_state.configs = [
            {"id": 1, "tau_mie": 0.084, "ozone": 340.0, "temp": 20.0, "rh": 50.0, "pwv_direct": 1.42,
             "use_config_spectral_elevation": False, "spectral_elevation": 30.0,
             "use_config_real_day": False, "real_day_date": datetime.date.today()}
        ]
        st.session_state.next_id = 2

    def add_config():
        st.session_state.configs.append({
            "id": st.session_state.next_id,
            "tau_mie": 0.084, "ozone": 340.0, "temp": 20.0, "rh": 50.0, "pwv_direct": 1.42,
            "use_config_spectral_elevation": False, "spectral_elevation": 30.0,
            "use_config_real_day": False, "real_day_date": datetime.date.today()
        })
        st.session_state.next_id += 1
        
    def remove_config(config_id):
        st.session_state.configs = [c for c in st.session_state.configs if c["id"] != config_id]

    st.button("Ajouter une configuration", on_click=add_config, key="add_config_btn")
    
    for i, c in enumerate(st.session_state.configs):
        with st.expander(f"Configuration {i+1}", expanded=True):
            cc1, cc2, cc3 = st.columns([1, 1, 0.5])
            with cc1:
                c["tau_mie"] = st.number_input("τ_Mie", value=float(c["tau_mie"]), key=f"tau_{c['id']}", format="%.3f", help="Profondeur optique des aérosols mesurée à 500 nm.")
            with cc2:
                c["ozone"] = st.number_input("Ozone (DU)", value=float(c["ozone"]), key=f"oz_{c['id']}", help="Colonne d'ozone en unités Dobson.")
            with cc3:
                st.markdown("<br>", unsafe_allow_html=True)
                st.button("Supprimer", key=f"rm_{c['id']}", on_click=remove_config, args=(c['id'],))
            
            calc_pwv_checked = st.checkbox(f"Calculer la PWV à partir de la temp. et de l'humidité", key=f"calc_pwv_{c['id']}")
            
            if calc_pwv_checked:
                temp_rh_col1, temp_rh_col2 = st.columns(2)
                with temp_rh_col1:
                    c["temp"] = st.number_input("Température (°C)", value=float(c.get("temp", 20.0)), key=f"tem_{c['id']}")
                with temp_rh_col2:
                    c["rh"] = st.number_input("Humidité relative (%)", value=float(c.get("rh", 50.0)), key=f"rh_{c['id']}")
                
                pwv_calc = calc_pwv(c["temp"], c["rh"])
                st.caption(f"*PWV calculée à partir de {c['temp']:.1f}°C et {c['rh']:.1f}% : **{pwv_calc:.3f} cm***")
                c['pwv_calc'] = pwv_calc
            else:
                if "pwv_direct" not in c: c["pwv_direct"] = 1.42
                c["pwv_direct"] = st.number_input(f"PWV (cm)", value=float(c["pwv_direct"]), key=f"pwv_direct_{c['id']}", format="%.3f")
                c['pwv_calc'] = c["pwv_direct"]
            
            if show_spectral_elevation_controls:
                st.markdown("---")
                c["use_config_spectral_elevation"] = st.checkbox(
                    "Utiliser une élévation spectrale spécifique à cette configuration",
                    value=bool(c.get("use_config_spectral_elevation", False)),
                    key=f"cfg_spec_elev_{c['id']}"
                )
                if c["use_config_spectral_elevation"]:
                    c["spectral_elevation"] = st.number_input(
                        "Élévation spectrale (°)",
                        value=float(c.get("spectral_elevation", 30.0)),
                        min_value=0.01,
                        max_value=90.0,
                        step=0.1,
                        format="%.2f",
                        key=f"cfg_spec_elev_value_{c['id']}"
                    )
            
            if show_real_day_controls:
                st.markdown("---")
                c["use_config_real_day"] = st.checkbox(
                    "Utiliser une journée réelle spécifique à cette configuration",
                    value=bool(c.get("use_config_real_day", False)),
                    key=f"cfg_real_day_{c['id']}"
                )
                if c["use_config_real_day"]:
                    c["real_day_date"] = st.date_input(
                        "Date de simulation",
                        value=c.get("real_day_date", datetime.date.today()),
                        key=f"cfg_real_day_date_{c['id']}"
                    )

def plot_and_download(fig, df, filename_prefix):
    """Save outputs: CSV and PNG"""
    col1, col2 = st.columns(2)
    with col1:
        try:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name=f'{filename_prefix}_donnees.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"Failed to generate CSV: {e}")
    
    with col2:
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label="Télécharger PNG",
                data=buf.getvalue(),
                file_name=f'{filename_prefix}_graphique.png',
                mime='image/png',
            )
        except Exception as e:
            st.error(f"Failed to generate image: {e}")

# Profil 1
if mode == "1. Profil unique":
    st.header("Explorateur de profil unique")
    st.markdown("Balayer les angles d'élévation pour une configuration atmosphérique fixe.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        tau_MIE = st.number_input("Profondeur optique des aérosols (AOD500)", value=0.084, format="%.3f",
                                  help="Profondeur optique des aérosols à 500 nm.")
    with col2:
        ozone_DU = st.number_input("Ozone (DU)", value=340.0, format="%.1f",
                                   help="Colonne d'ozone en unités Dobson.")
    with col3:
        pwv_CM = st.number_input("PWV (cm)", value=1.42, format="%.3f",
                                 help="Valeur directe de vapeur d'eau précipitable en cm")
        calc_pwv_checked = st.checkbox("Calculer la PWV à partir de la temp. et de l'humidité")

    if calc_pwv_checked:
        temp_rh_col1, temp_rh_col2 = st.columns(2)
        with temp_rh_col1:
            temp_C = st.number_input("Température (°C)", value=20.0, format="%.1f")
        with temp_rh_col2:
            rh_PCT = st.number_input("Humidité relative (%)", value=50.0, format="%.1f")
        pwv_CM = calc_pwv(temp_C, rh_PCT)
        st.info(f"PWV calculée à partir de Temp={temp_C}°C et HR={rh_PCT}% : **{pwv_CM:.3f} cm**")
        
    st.subheader("Options d'affichage")
    c1, c2, c3 = st.columns(3)
    with c1: plot_dni = st.checkbox("Tracer le rayonnement direct normal (DNI)", value=True)
    with c2: plot_dhi = st.checkbox("Tracer le rayonnement diffus horizontal (DHI)", value=True)
    with c3: plot_ghi = st.checkbox("Tracer le rayonnement global horizontal (GHI)", value=True)
    
    # We remove the button so the graphs respond dynamically
    tau_O3, tau_H2O = obtenir_epaisseurs_optiques_absorption(ozone_DU, pwv_CM)
        
    elevations = np.round(np.arange(0.01, 90.01, 0.01), 2)  # 0.01 degree steps
    records = []
    
    for i, elev in enumerate(elevations):
        theta_z = 90.0 - elev
        _, _, _, DNI, DHI, GHI = calc_irradiance(
            I0=I0, tau_R=tau_R, tau_O3=tau_O3, tau_H2O=tau_H2O,
            zenith_deg=theta_z, tau_mie=tau_MIE, debug=False
        )
        records.append({
            'Élévation (deg)': elev, 'Zénith (deg)': theta_z, 
            'Rayonnement direct normal (W/m²)': DNI,
            'Rayonnement diffus horizontal (W/m²)': DHI,
            'Rayonnement global horizontal (W/m²)': GHI
        })
        
    df = pd.DataFrame(records)
        
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    if plot_dni: ax.plot(df['Élévation (deg)'], df['Rayonnement direct normal (W/m²)'], label='DNI')
    if plot_dhi: ax.plot(df['Élévation (deg)'], df['Rayonnement diffus horizontal (W/m²)'], label='DHI')
    if plot_ghi: ax.plot(df['Élévation (deg)'], df['Rayonnement global horizontal (W/m²)'], label='GHI')
    
    ax.set_xlabel('Angle d'"'"'élévation (degrés)')
    ax.set_ylabel('Irradiance (W/m²)')
    ax.set_title("Irradiance en fonction de l'angle d'élévation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)
    st.dataframe(df)
    plot_and_download(fig, df, "profil_unique")

# MODE 2: Analyse avancee
elif mode == "2. Analyse avancée (comparaison / journée réelle)":
    st.header("Analyse avancée")
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis_mode = st.radio("Mode d'axe horizontal", ["Angle d'élévation (0-90°)", "Journée réelle (heure locale)"], horizontal=True)
    
    if x_axis_mode == "Journée réelle (heure locale)":
        st.subheader("Lieu et date")
        coord_col1, coord_col2, coord_col3 = st.columns(3)
        with coord_col1:
            latitude = st.number_input("Latitude", value=45.5019, format="%.4f")
        with coord_col2:
            longitude = st.number_input("Longitude", value=-73.5674, format="%.4f")
        with coord_col3:
            date_sel = st.date_input("Date", datetime.date.today())
            
    st.markdown("---")
    render_configuration_ui(show_real_day_controls=(x_axis_mode == "Journée réelle (heure locale)"))

    metric_label_to_code = {
        "Rayonnement global horizontal (GHI)": "GHI",
        "Rayonnement direct normal (DNI)": "DNI",
        "Rayonnement diffus horizontal (DHI)": "DHI",
    }
    metric = st.selectbox("Métrique à comparer", list(metric_label_to_code.keys()))
    metric_code = metric_label_to_code[metric]

    st.markdown("---")
    if st.button("Générer l'analyse"):
        if not st.session_state.configs:
            st.warning("Veuillez ajouter au moins une configuration.")
            st.stop()
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if x_axis_mode == "Angle d'élévation (0-90°)":
            elevations = np.round(np.arange(0.01, 90.01, 0.01), 2)
            all_data = pd.DataFrame({'Élévation (deg)': elevations})
            
            prog_bar = st.progress(0, text="Calcul du balayage en élévation...")
            
            for i, c in enumerate(st.session_state.configs):
                tau_O3, tau_H2O = obtenir_epaisseurs_optiques_absorption(c["ozone"], c["pwv_calc"])
                vals = []
                for j, elev in enumerate(elevations):
                    theta_z = 90.0 - elev
                    _, _, _, DNI, DHI, GHI = calc_irradiance(
                        I0=I0, tau_R=tau_R, tau_O3=tau_O3, tau_H2O=tau_H2O,
                        zenith_deg=theta_z, tau_mie=c["tau_mie"], debug=False
                    )
                    if metric_code == "GHI": vals.append(GHI)
                    elif metric_code == "DNI": vals.append(DNI)
                    elif metric_code == "DHI": vals.append(DHI)
                    
                    if j % 20 == 0:
                        prog_bar.progress((i + j/len(elevations))/len(st.session_state.configs), text=f"Calcul de la configuration {i+1}...")
                        
                label = f"Config {i+1} (τ_Mie={c['tau_mie']:.3f}, DU={c['ozone']}, PWV={c['pwv_calc']:.2f})"
                ax.plot(elevations, vals, label=label)
                col_name = f"Config{i+1}_τMie{c['tau_mie']:.3f}_O3{c['ozone']:.0f}DU_PWV{c['pwv_calc']:.2f}cm_{metric_code}"
                all_data[col_name] = vals
                
            prog_bar.empty()
            ax.set_xlabel("Angle d'élévation (degrés)")
            
        else:
            tf = TimezoneFinder()
            tz_str = tf.timezone_at(lng=longitude, lat=latitude)
            if tz_str is None:
                st.warning("Impossible de déterminer le fuseau horaire pour ces coordonnées. UTC sera utilisé par défaut.")
                tz = pytz.UTC
            else:
                tz = pytz.timezone(tz_str)
                
            plot_start = tz.localize(datetime.datetime.combine(date_sel, datetime.time(0, 0)))
            plot_end = tz.localize(datetime.datetime.combine(date_sel, datetime.time(23, 59)))
            plot_times = pd.date_range(start=plot_start, end=plot_end, freq='1min')
            all_data = pd.DataFrame({'Heure': [t.strftime("%H:%M") for t in plot_times]})
            
            prog_bar = st.progress(0, text="Calcul de la simulation d'une journée réelle (intervalles d'une minute)...")
            
            for i, c in enumerate(st.session_state.configs):
                config_date = c["real_day_date"] if c.get("use_config_real_day", False) else date_sel
                start_time = tz.localize(datetime.datetime.combine(config_date, datetime.time(0, 0)))
                end_time = tz.localize(datetime.datetime.combine(config_date, datetime.time(23, 59)))
                times = pd.date_range(start=start_time, end=end_time, freq='1min')
                solar_positions = pvlib.solarposition.get_solarposition(times, latitude, longitude)
                tau_O3, tau_H2O = obtenir_epaisseurs_optiques_absorption(c["ozone"], c["pwv_calc"])
                vals = []
                
                for j, (t, row) in enumerate(zip(times, solar_positions.itertuples())):
                    elev = row.elevation
                    if elev > 0:
                        theta_z = 90.0 - elev
                        _, _, _, DNI, DHI, GHI = calc_irradiance(
                            I0=I0, tau_R=tau_R, tau_O3=tau_O3, tau_H2O=tau_H2O,
                            zenith_deg=theta_z, tau_mie=c["tau_mie"], debug=False
                        )
                        if metric_code == "GHI": vals.append(GHI)
                        elif metric_code == "DNI": vals.append(DNI)
                        elif metric_code == "DHI": vals.append(DHI)
                    else:
                        vals.append(0.0)
                        
                    if j % 50 == 0:
                        prog_bar.progress((i + j/len(times))/len(st.session_state.configs), text=f"Calcul de la configuration {i+1}...")
                        
                label = f"Config {i+1} (τ_Mie={c['tau_mie']:.3f}, DU={c['ozone']}, PWV={c['pwv_calc']:.2f})"
                
                vals_np = np.array(vals)
                sun_up_mask = solar_positions['elevation'] > 0
                
                ax.plot(plot_times[sun_up_mask], vals_np[sun_up_mask], label=f"{label} | {config_date.isoformat()}")
                col_name = f"Config{i+1}_τMie{c['tau_mie']:.3f}_O3{c['ozone']:.0f}DU_PWV{c['pwv_calc']:.2f}cm_{metric_code}"
                all_data[col_name] = vals
                    
            prog_bar.empty()
            from matplotlib.dates import DateFormatter
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M', tz=tz))
            ax.set_xlabel(f"Heure locale ({tz_str})")
            plt.xticks(rotation=45)
            
        ax.set_ylabel(f'{metric} (W/m²)')
        ax.set_title(f'Comparaison de {metric}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        
        st.pyplot(fig)
        st.dataframe(all_data)
        plot_and_download(fig, all_data, "analyse_comparative")

# analyse spectrale (MODE 3)
elif mode == "3. Analyse spectrale":
    st.header("Analyse spectrale")
    st.markdown("Comparer le GHI spectral entre plusieurs configurations personnalisées.")
    
    render_configuration_ui(show_spectral_elevation_controls=True)
    all_data = pd.DataFrame({'Longueur d'"'"'onde (nm)': LONGUEURS_ONDE_NM})
    
    st.markdown("---")
    st.subheader("Détails de l'analyse spectrale")
    #st.markdown("Comparer le GHI spectral entre plusieurs configurations sur une surface horizontale. Chaque configuration peut utiliser indépendamment le modèle physique (Beer-Lambert) et/ou le modèle PVLib (SPECTRL2).")
    
    col_spec_1 = st.columns(1)[0]
    with col_spec_1:
        elev_spectral = st.slider("Angle d'élévation solaire (°)", min_value=0.01, max_value=90.0, value=30.0, step=0.1,
                                        help="Angle d'élévation solaire pour la comparaison spectrale (0,01° - 90,0°)")
    
    zenith_spectral = 90.0 - elev_spectral
    
    if not st.session_state.configs:
        st.warning("Veuillez ajouter au moins une configuration.")
        st.stop()

    try:
        fig_comp, ax_comp = plt.subplots(figsize=(14, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, len(st.session_state.configs)))
        
        for i, c in enumerate(st.session_state.configs):
            ozone_atm_cm_config = c["ozone"] * 0.001
            elev_for_config = c["spectral_elevation"] if c.get("use_config_spectral_elevation", False) else elev_spectral
            zenith_for_config = 90.0 - elev_for_config
            
            # --- DEBUT ANCIEN MODELE PHYSIQUE ---
            if c.get("use_physics", True):
                tau_O3_cfg, tau_H2O_cfg = obtenir_epaisseurs_optiques_absorption(c["ozone"], c["pwv_calc"])
                _, _, GHI_spectral, _, _, _ = calc_irradiance(
                    I0=I0, tau_R=tau_R, tau_O3=tau_O3_cfg, tau_H2O=tau_H2O_cfg,
                    zenith_deg=zenith_for_config, tau_mie=c["tau_mie"], debug=False
                )
                label_physics = f"Config {i+1} - Physique (τ_Mie={c['tau_mie']:.3f}, O₃={c['ozone']:.0f}DU, PWV={c['pwv_calc']:.2f}cm, Élév={elev_for_config:.1f}°)"
                ax_comp.plot(LONGUEURS_ONDE_NM, GHI_spectral, linewidth=2, label=label_physics, color=colors[i], linestyle='-')
                all_data[f"Config_{i+1}_Physique"] = GHI_spectral
            
            # --- FIN ANCIEN MODELE PHYSIQUE ---
            
            #if c.get("use_spectrl2", True):
#                wl_s2, ghi_s2 = compute_spectrl2_spectrum(
#                    zenith_angle_deg=zenith_for_config,
#                    aod500=c["tau_mie"],
#                    ozone_atm_cm=ozone_atm_cm_config,
#                    pwv_cm=c["pwv_calc"]
#                )
#                label_pvlib = f"Config {i+1} - PVLib (τ_Mie={c['tau_mie']:.3f}, O₃={c['ozone']:.0f}DU, PWV={c['pwv_calc']:.2f}cm, Élév={elev_for_config:.1f}°)"
#                ax_comp.plot(wl_s2, ghi_s2, linewidth=2, label=label_pvlib, color=colors[i], linestyle='--')
        
        ax_comp.set_xlabel('Longueur d'"'"'onde (nm)', fontsize=11)
        ax_comp.set_ylabel('Irradiance spectrale (W/m²/nm)', fontsize=11)
        ax_comp.set_title(f"Comparaison du GHI spectral - Élévation {elev_spectral:.1f}°", fontsize=12)
        ax_comp.grid(True, alpha=0.3)
        ax_comp.legend(fontsize=9, loc='upper right')
        ax_comp.set_xlim(100, 4000)
        ax_comp.set_ylim(bottom=0)
        
        st.pyplot(fig_comp)
        st.dataframe(all_data)
        plot_and_download(fig_comp, all_data, "analyse_comparative")
        
    except Exception as e:
        st.error(f"Erreur lors du calcul de la comparaison spectrale : {e}")
