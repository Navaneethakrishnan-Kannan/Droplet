import streamlit as st
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

# --- Constants and Look-up Data (derived from the article) ---

# Table 1: Typical Liquid Surface Tension Values (dyne/cm)
SURFACE_TENSION_TABLE = {
    "Water/gas": 72,
    "Light crude oil/gas": 32,
    "Heavy crude oil/gas": 37,
    "Condensate/gas": 25,
    "Liquid petroleum gas/gas": 10,
    "Natural gas liquids (high C2)/gas": 5,
    "Triethylene glycol": 45,
    "Amine solutions": 52.5, # Average of 45-60 from the range provided
}
# Conversion: 1 dyne/cm = 0.0022 poundal/ft (from the article)
DYNE_CM_TO_POUNDAL_FT = 0.0022

# Typical values for Upper-Limit Log Normal Distribution (from the article)
A_DISTRIBUTION = 4.0
DELTA_DISTRIBUTION = 0.72

# Figure 9 Approximation for Droplet Size Distribution Shift Factor
def get_shift_factor(inlet_device, rho_v_squared):
    """
    Approximates the droplet size distribution shift factor based on the inlet device
    and inlet momentum (rho_g * V_g^2).
    """
    if inlet_device == "No inlet device":
        return 1.0
    elif inlet_device == "Diverter plate":
        if rho_v_squared <= 1000: return 1.0 - (0.1 * rho_v_squared / 1000)
        elif rho_v_squared <= 2000: return 0.9 - (0.3 * (rho_v_squared - 1000) / 1000)
        elif rho_v_squared <= 3000: return 0.6 - (0.4 * (rho_v_squared - 2000) / 1000)
        else: return 0.2
    elif inlet_device == "Half-pipe":
        if rho_v_squared <= 1000: return 1.0 - (0.1 * rho_v_squared / 1000)
        elif rho_v_squared <= 2000: return 0.9 - (0.2 * (rho_v_squared - 1000) / 1000)
        elif rho_v_squared <= 3000: return 0.7 - (0.2 * (rho_v_squared - 2000) / 1000)
        elif rho_v_squared <= 4000: return 0.5 - (0.2 * (rho_v_squared - 3000) / 1000)
        else: return 0.2
    elif inlet_device == "Vane-type":
        if rho_v_squared <= 5000: return 1.0 - (0.05 * rho_v_squared / 5000)
        elif rho_v_squared <= 7000: return 0.95 - (0.15 * (rho_v_squared - 5000) / 2000)
        elif rho_v_squared <= 10000: return 0.8 - (0.3 * (rho_v_squared - 7000) / 3000)
        else: return 0.5
    elif inlet_device == "Cyclonic":
        if rho_v_squared <= 10000: return 1.0 - (0.1 * rho_v_squared / 10000)
        elif rho_v_squared <= 14000: return 0.9 - (0.7 * (rho_v_squared - 10000) / 4000)
        else: return 0.2
    return 1.0

# --- Unit Conversion Factors ---
M_TO_FT = 3.28084 # 1 meter = 3.28084 feet
KG_TO_LB = 2.20462 # 1 kg = 2.20462 lb
MPS_TO_FTPS = 3.28084 # 1 m/s = 3.28084 ft/s
PAS_TO_LB_FT_S = 0.67197 # 1 Pa.s (kg/m.s) = 0.67197 lb/ft.s
KG_M3_TO_LB_FT3 = 0.0624279 # 1 kg/m^3 = 0.0624279 lb/ft^3
NM_TO_POUNDAL_FT = 2.2 # 1 N/m = 1000 dyne/cm; 1 dyne/cm = 0.0022 poundal/ft => 1 N/m = 2.2 poundal/ft

MICRON_TO_FT = 1e-6 * M_TO_FT
FT_TO_MICRON = 1 / MICRON_TO_FT

def convert_value(value, unit_type, from_system, to_system):
    """Converts a value between FPS and SI units for a given unit type."""
    if from_system == to_system:
        return value

    if from_system == "FPS" and to_system == "SI":
        if unit_type == "length": return value / M_TO_FT
        elif unit_type == "velocity": return value / MPS_TO_FTPS
        elif unit_type == "density": return value / KG_M3_TO_LB_FT3
        elif unit_type == "viscosity": return value / PAS_TO_LB_FT_S
        elif unit_type == "surface_tension": return value / NM_TO_POUNDAL_FT # poundal/ft to N/m
        elif unit_type == "momentum": return value / 1.48816 # lb/ft-s^2 to Pa
    elif from_system == "SI" and to_system == "FPS":
        if unit_type == "length": return value * M_TO_FT
        elif unit_type == "velocity": return value * MPS_TO_FTPS
        elif unit_type == "density": return value * KG_M3_TO_LB_FT3
        elif unit_type == "viscosity": return value * PAS_TO_LB_FT_S
        elif unit_type == "surface_tension": return value * NM_TO_POUNDAL_FT # N/m to poundal/ft
        elif unit_type == "momentum": return value * 1.48816 # Pa to lb/ft-s^2
    return value

def on_unit_system_change():
    """Callback function to handle unit system changes and convert input values."""
    current_system = st.session_state.unit_system # This is the NEW system
    previous_system = st.session_state.previous_unit_system # This is the OLD system

    if current_system != previous_system:
        # Convert all stored inputs from previous_system to current_system
        input_keys = ['D_input', 'rho_l_input', 'mu_l_input', 'V_g_input', 'rho_g_input', 'mu_g_input']
        unit_types = ['length', 'density', 'viscosity', 'velocity', 'density', 'viscosity']

        for key, unit_type in zip(input_keys, unit_types):
            if key in st.session_state.inputs:
                st.session_state.inputs[key] = convert_value(st.session_state.inputs[key], unit_type, previous_system, current_system)
        
        # Handle custom surface tension conversion
        if st.session_state.inputs['surface_tension_option'] == "Custom" and 'sigma_custom' in st.session_state.inputs:
            st.session_state.inputs['sigma_custom'] = convert_value(st.session_state.inputs['sigma_custom'], "surface_tension", previous_system, current_system)
        
        # Re-calculate sigma_fps based on the new unit system and potentially converted custom value
        sigma_input_val_after_conversion = 0.0
        if st.session_state.inputs['surface_tension_option'] == "Custom":
            sigma_input_val_after_conversion = st.session_state.inputs['sigma_custom']
        else:
            sigma_input_val_after_conversion = SURFACE_TENSION_TABLE[st.session_state.inputs['surface_tension_option']]
            if current_system == "SI":
                sigma_input_val_after_conversion *= 0.001 # Convert dyne/cm to N/m for SI display

        if current_system == "FPS":
            st.session_state.inputs['sigma_fps'] = sigma_input_val_after_conversion * DYNE_CM_TO_POUNDAL_FT
        else: # SI
            st.session_state.inputs['sigma_fps'] = sigma_input_val_after_conversion * NM_TO_POUNDAL_FT

    # Update previous_unit_system for the next change
    st.session_state.previous_unit_system = current_system

# --- Streamlit App Layout ---

st.set_page_config(layout="centered", page_title="Oil & Gas Separation App")

st.title("🛢️ Oil and Gas Separation: Particle Size Distribution")
st.markdown("""
This application helps quantify the entrained liquid droplet size distribution in gas-liquid separators,
based on the principles and correlations discussed in the article "Quantifying Separation Performance" by Mark Bothamley.
""")

# Initialize session state for inputs and results if not already present
if 'unit_system' not in st.session_state:
    st.session_state.unit_system = "FPS"
if 'inputs' not in st.session_state:
    # Initialize inputs with default FPS values
    st.session_state.inputs = {
        'D_input': 1.0, # ft
        'rho_l_input': 40.0, # lb/ft3
        'mu_l_input': 0.0005, # lb/ft-sec
        'V_g_input': 50.0, # ft/sec
        'rho_g_input': 0.1, # lb/ft3
        'mu_g_input': 0.00001, # lb/ft-sec
        'surface_tension_option': "Water/gas",
        'sigma_custom': 30.0, # dyne/cm (FPS default for custom)
        'inlet_device': "No inlet device",
    }
    # Initialize sigma_fps based on the initial default surface tension option
    st.session_state.inputs['sigma_fps'] = SURFACE_TENSION_TABLE["Water/gas"] * DYNE_CM_TO_POUNDAL_FT

if 'previous_unit_system' not in st.session_state:
    st.session_state.previous_unit_system = st.session_state.unit_system

if 'calculation_results' not in st.session_state:
    st.session_state.calculation_results = None
if 'plot_data' not in st.session_state:
    st.session_state.plot_data = None

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Input Parameters", "Calculation Steps", "Droplet Distribution Results"])

# --- Page: Input Parameters ---
if page == "Input Parameters":
    st.header("1. Input Parameters")

    # The radio button for unit system
    st.session_state.unit_system = st.radio(
        "Select Unit System",
        ["FPS", "SI"],
        key='unit_selection_radio',
        on_change=on_unit_system_change, # This callback handles the conversion
        help="Choose between Foot-Pound-Second (FPS) or International System (SI) units."
    )

    # Define unit labels based on current session state unit system
    if st.session_state.unit_system == "FPS":
        len_unit = "ft"
        dens_unit = "lb/ft³"
        vel_unit = "ft/sec"
        visc_unit = "lb/ft-sec"
        surf_tens_input_unit = "dyne/cm"
    else: # SI
        len_unit = "m"
        dens_unit = "kg/m³"
        vel_unit = "m/s"
        visc_unit = "Pa·s"
        surf_tens_input_unit = "N/m"

    st.subheader(f"Feed Pipe Conditions ({st.session_state.unit_system} Units)")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.inputs['D_input'] = st.number_input(f"Pipe Inside Diameter ({len_unit})", min_value=0.01, value=st.session_state.inputs['D_input'], format="%.2f", key='D_input_widget',
                            help="Diameter of the feed pipe to the separator.")
        st.session_state.inputs['rho_l_input'] = st.number_input(f"Liquid Density ({dens_unit})", min_value=0.01, value=st.session_state.inputs['rho_l_input'], format="%.2f", key='rho_l_input_widget',
                                help="Density of the liquid phase.")
        st.session_state.inputs['mu_l_input'] = st.number_input(f"Liquid Viscosity ({visc_unit})", min_value=1e-7, value=st.session_state.inputs['mu_l_input'], format="%.7f", key='mu_l_input_widget',
                                help=f"Viscosity of the liquid phase. Example: Water at 60°F is ~0.00075 {visc_unit} (FPS) or ~0.0011 {visc_unit} (SI).")
    with col2:
        st.session_state.inputs['V_g_input'] = st.number_input(f"Gas Velocity ({vel_unit})", min_value=0.01, value=st.session_state.inputs['V_g_input'], format="%.2f", key='V_g_input_widget',
                              help="Superficial gas velocity in the feed pipe.")
        st.session_state.inputs['rho_g_input'] = st.number_input(f"Gas Density ({dens_unit})", min_value=1e-4, value=st.session_state.inputs['rho_g_input'], format="%.4f", key='rho_g_input_widget',
                                help="Density of the gas phase.")
        st.session_state.inputs['mu_g_input'] = st.number_input(f"Gas Viscosity ({visc_unit})", min_value=1e-8, value=st.session_state.inputs['mu_g_input'], format="%.8f", key='mu_g_input_widget',
                                help=f"Viscosity of the gas phase. Example: Methane at 60°F is ~0.000007 {visc_unit} (FPS) or ~0.00001 {visc_unit} (SI).")

    st.markdown("---")
    col_st, col_id = st.columns(2)

    with col_st:
        st.subheader("Liquid Surface Tension")
        # Ensure the selectbox reflects the current state
        current_st_option_index = list(SURFACE_TENSION_TABLE.keys()).index(st.session_state.inputs['surface_tension_option']) if st.session_state.inputs['surface_tension_option'] in SURFACE_TENSION_TABLE else len(SURFACE_TENSION_TABLE.keys())
        st.session_state.inputs['surface_tension_option'] = st.selectbox(
            "Select Fluid System for Surface Tension",
            options=list(SURFACE_TENSION_TABLE.keys()) + ["Custom"],
            index=current_st_option_index,
            key='surface_tension_option_select',
            help="Choose a typical value or enter a custom one."
        )

        sigma_input_val = 0.0
        if st.session_state.inputs['surface_tension_option'] == "Custom":
            st.session_state.inputs['sigma_custom'] = st.number_input(f"Custom Liquid Surface Tension ({surf_tens_input_unit})", min_value=0.01, value=st.session_state.inputs['sigma_custom'], format="%.2f", key='sigma_custom_input')
            sigma_input_val = st.session_state.inputs['sigma_custom']
        else:
            sigma_input_val = SURFACE_TENSION_TABLE[st.session_state.inputs['surface_tension_option']]
            if st.session_state.unit_system == "SI":
                sigma_input_val = sigma_input_val * 0.001 # 1 dyne/cm = 0.001 N/m

        # This part ensures sigma_fps is always up-to-date based on the current inputs and unit system
        if st.session_state.unit_system == "FPS":
            st.session_state.inputs['sigma_fps'] = sigma_input_val * DYNE_CM_TO_POUNDAL_FT
        else:
            st.session_state.inputs['sigma_fps'] = sigma_input_val * NM_TO_POUNDAL_FT
        
        st.info(f"**Calculated Liquid Surface Tension (for calculation):** {st.session_state.inputs['sigma_fps']:.4f} poundal/ft")


    with col_id:
        st.subheader("Separator Inlet Device")
        # Ensure the selectbox reflects the current state
        current_inlet_device_index = ["No inlet device", "Diverter plate", "Half-pipe", "Vane-type", "Cyclonic"].index(st.session_state.inputs['inlet_device'])
        st.session_state.inputs['inlet_device'] = st.selectbox(
            "Choose Inlet Device Type",
            options=["No inlet device", "Diverter plate", "Half-pipe", "Vane-type", "Cyclonic"],
            index=current_inlet_device_index,
            key='inlet_device_select',
            help="The inlet device influences the droplet size distribution downstream."
        )

    st.markdown("---")

    if st.button("Calculate Particle Size Distribution", help="Click to perform calculations and view results."):
        try:
            # Convert all inputs to FPS for consistent calculation
            D = convert_value(st.session_state.inputs['D_input'], "length", st.session_state.unit_system, "FPS")
            rho_l = convert_value(st.session_state.inputs['rho_l_input'], "density", st.session_state.unit_system, "FPS")
            mu_l = convert_value(st.session_state.inputs['mu_l_input'], "viscosity", st.session_state.unit_system, "FPS")
            V_g = convert_value(st.session_state.inputs['V_g_input'], "velocity", st.session_state.unit_system, "FPS")
            rho_g = convert_value(st.session_state.inputs['rho_g_input'], "density", st.session_state.unit_system, "FPS")
            mu_g = convert_value(st.session_state.inputs['mu_g_input'], "viscosity", st.session_state.unit_system, "FPS")
            sigma = st.session_state.inputs['sigma_fps'] # This is already in poundal/ft

            # --- Perform Calculations ---
            results = {}
            plot_data = {}

            # Step 1: Calculate Superficial Gas Reynolds Number (Re_g)
            if mu_g == 0: raise ValueError("Gas viscosity (μg) cannot be zero for Reynolds number calculation.")
            Re_g = (D * V_g * rho_g) / mu_g
            results['Re_g'] = Re_g

            # Step 2: Calculate Volume Median Diameter ($d_{v50}$) without inlet device effect
            if V_g == 0 or rho_g == 0 or rho_l == 0 or mu_l == 0:
                raise ValueError("Gas velocity, gas density, liquid density, and liquid viscosity must be non-zero for $d_{v50}$ calculation.")
            
            dv50_original_fps = 0.01 * (sigma / (rho_g * V_g**2)) * (Re_g**2) * ((rho_g / rho_l)**(-1/3)) * ((mu_g / mu_l)**(2/3))
            results['dv50_original_fps'] = dv50_original_fps

            # Step 3: Determine Inlet Momentum (rho_g V_g^2)
            rho_v_squared_fps = rho_g * V_g**2
            results['rho_v_squared_fps'] = rho_v_squared_fps

            # Step 4: Apply Inlet Device "Droplet Size Distribution Shift Factor"
            shift_factor = get_shift_factor(st.session_state.inputs['inlet_device'], rho_v_squared_fps)
            dv50_adjusted_fps = dv50_original_fps * shift_factor
            results['shift_factor'] = shift_factor
            results['dv50_adjusted_fps'] = dv50_adjusted_fps

            # Step 5: Calculate parameters for Upper-Limit Log Normal Distribution
            d_max_fps = A_DISTRIBUTION * dv50_adjusted_fps
            results['d_max_fps'] = d_max_fps

            # Step 6: Generate Volume Fraction and Cumulative Volume Fraction for a range of droplet sizes
            dp_min_calc_fps = dv50_adjusted_fps * 0.01
            dp_max_calc_fps = d_max_fps * 0.999

            dp_values_ft = np.linspace(dp_min_calc_fps, dp_max_calc_fps, 500)
            
            volume_fraction = []
            cumulative_volume_undersize = []

            for dp in dp_values_ft:
                z_val = np.log((A_DISTRIBUTION * dp) / (d_max_fps - dp))
                fv_dp = (DELTA_DISTRIBUTION * d_max_fps) / np.sqrt(np.pi * dp * (d_max_fps - dp)) * np.exp(-DELTA_DISTRIBUTION**2 * z_val**2)
                volume_fraction.append(fv_dp)
                v_under = 1 - 0.5 * (1 - erf(DELTA_DISTRIBUTION * z_val))
                cumulative_volume_undersize.append(v_under)
            
            plot_data['dp_values_ft'] = dp_values_ft
            plot_data['volume_fraction'] = volume_fraction
            plot_data['cumulative_volume_undersize'] = cumulative_volume_undersize

            st.session_state.calculation_results = results
            st.session_state.plot_data = plot_data
            st.sidebar.success("Calculations complete! Navigate to other pages.")
            st.success("Calculations complete! You can now view the results in the 'Calculation Steps' and 'Droplet Distribution Results' pages.")

        except Exception as e:
            st.error(f"An error occurred during calculation: {e}")
            st.session_state.calculation_results = None
            st.session_state.plot_data = None


# --- Page: Calculation Steps ---
elif page == "Calculation Steps":
    st.header("2. Step-by-Step Calculation Results")

    if st.session_state.calculation_results:
        results = st.session_state.calculation_results
        unit_system = st.session_state.unit_system

        # Define unit labels for display
        if unit_system == "FPS":
            len_unit = "ft"
            dens_unit = "lb/ft³"
            vel_unit = "ft/sec"
            visc_unit = "lb/ft-sec"
            momentum_unit = "lb/ft-sec²"
            micron_unit_label = "microns"
        else: # SI
            len_unit = "m"
            dens_unit = "kg/m³"
            vel_unit = "m/s"
            visc_unit = "Pa·s"
            momentum_unit = "Pa"
            micron_unit_label = "µm"

        # Display inputs used for calculation (converted to FPS)
        st.subheader("Inputs Used for Calculation (Converted to FPS)")
        st.write(f"Pipe Inside Diameter (D): {convert_value(st.session_state.inputs['D_input'], 'length', unit_system, 'FPS'):.2f} ft")
        st.write(f"Liquid Density (ρl): {convert_value(st.session_state.inputs['rho_l_input'], 'density', unit_system, 'FPS'):.2f} lb/ft³")
        st.write(f"Liquid Viscosity (μl): {convert_value(st.session_state.inputs['mu_l_input'], 'viscosity', unit_system, 'FPS'):.7f} lb/ft-sec")
        st.write(f"Gas Velocity (Vg): {convert_value(st.session_state.inputs['V_g_input'], 'velocity', unit_system, 'FPS'):.2f} ft/sec")
        st.write(f"Gas Density (ρg): {convert_value(st.session_state.inputs['rho_g_input'], 'density', unit_system, 'FPS'):.4f} lb/ft³")
        st.write(f"Gas Viscosity (μg): {convert_value(st.session_state.inputs['mu_g_input'], 'viscosity', unit_system, 'FPS'):.8f} lb/ft-sec")
        st.write(f"Liquid Surface Tension (σ): {st.session_state.inputs['sigma_fps']:.4f} poundal/ft")
        st.write(f"Selected Inlet Device: {st.session_state.inputs['inlet_device']}")
        st.markdown("---")

        # Step 1: Calculate Superficial Gas Reynolds Number (Re_g)
        st.markdown("#### Step 1: Calculate Superficial Gas Reynolds Number ($Re_g$)")
        D_fps = convert_value(st.session_state.inputs['D_input'], "length", unit_system, "FPS")
        V_g_fps = convert_value(st.session_state.inputs['V_g_input'], "velocity", unit_system, "FPS")
        rho_g_fps = convert_value(st.session_state.inputs['rho_g_input'], "density", unit_system, "FPS")
        mu_g_fps = convert_value(st.session_state.inputs['mu_g_input'], "viscosity", unit_system, "FPS")

        st.write(f"Equation: $Re_g = \\frac{{D \\cdot V_g \\cdot \\rho_g}}{{\\mu_g}}$")
        st.write(f"Calculation: $Re_g = \\frac{{{D_fps:.2f} \\text{{ ft}} \\cdot {V_g_fps:.2f} \\text{{ ft/sec}} \\cdot {rho_g_fps:.4f} \\text{{ lb/ft}}^3}}{{{mu_g_fps:.8f} \\text{{ lb/ft-sec}}}} = {results['Re_g']:.2f}$")
        st.success(f"**Result:** Superficial Gas Reynolds Number ($Re_g$) = **{results['Re_g']:.2f}** (dimensionless)")

        st.markdown("---")

        # Step 2: Calculate Volume Median Diameter ($d_{v50}$) without inlet device effect
        st.markdown("#### Step 2: Calculate Initial Volume Median Diameter ($d_{v50}$) (Kataoka et al., 1983)")
        rho_l_fps = convert_value(st.session_state.inputs['rho_l_input'], "density", unit_system, "FPS")
        mu_l_fps = convert_value(st.session_state.inputs['mu_l_input'], "viscosity", unit_system, "FPS")

        dv50_original_display = from_fps(results['dv50_original_fps'], "length", unit_system)
        
        st.write(f"Equation: $d_{{v50}} = 0.01 \\left(\\frac{{\\sigma}}{{\\rho_g V_g^2}}\\right) Re_g^2 \\left(\\frac{{\\rho_g}}{{\\rho_l}}\\right)^{{-1/3}} \\left(\\frac{{\\mu_g}}{{\\mu_l}}\\right)^{{2/3}}$")
        st.write(f"Calculation: $d_{{v50}} = 0.01 \\left(\\frac{{{st.session_state.inputs['sigma_fps']:.4f}}}{{{rho_g_fps:.4f} \\cdot ({V_g_fps:.2f})^2}}\\right) ({results['Re_g']:.2f})^2 \\left(\\frac{{{rho_g_fps:.4f}}}{{{rho_l_fps:.2f}}}\\right)^{{-0.333}} \\left(\\frac{{{mu_g_fps:.8f}}}{{{mu_l_fps:.7f}}}\\right)^{{0.667}}$")
        st.success(f"**Result:** Initial Volume Median Diameter ($d_{{v50}}$) = **{results['dv50_original_fps'] * FT_TO_MICRON:.2f} {micron_unit_label}** ({dv50_original_display:.6f} {len_unit})")

        st.markdown("---")

        # Step 3: Determine Inlet Momentum (rho_g V_g^2)
        st.markdown("#### Step 3: Calculate Inlet Momentum ($\\rho_g V_g^2$)")
        rho_v_squared_display = from_fps(results['rho_v_squared_fps'], "momentum", unit_system)
        st.write(f"Equation: $\\rho_g V_g^2 = \\rho_g \\cdot V_g^2$")
        st.write(f"Calculation: $\\rho_g V_g^2 = {rho_g_fps:.4f} \\text{{ lb/ft}}^3 \\cdot ({V_g_fps:.2f} \\text{{ ft/sec}})^2 = {results['rho_v_squared_fps']:.2f} \\text{{ lb/ft-sec}}^2$")
        st.success(f"**Result:** Inlet Momentum ($\\rho_g V_g^2$) = **{rho_v_squared_display:.2f} {momentum_unit}**")

        st.markdown("---")

        # Step 4: Apply Inlet Device "Droplet Size Distribution Shift Factor"
        st.markdown("#### Step 4: Apply Inlet Device Effect (Droplet Size Distribution Shift Factor)")
        st.write(f"Selected Inlet Device: **{st.session_state.inputs['inlet_device']}**")
        dv50_adjusted_display = from_fps(results['dv50_adjusted_fps'], "length", unit_system)
        st.write(f"Based on Figure 9 from the article, for an inlet momentum of {rho_v_squared_display:.2f} {momentum_unit} and a '{st.session_state.inputs['inlet_device']}' device, the estimated shift factor is **{results['shift_factor']:.3f}**.")
        st.write(f"Equation: $d_{{v50, adjusted}} = d_{{v50, original}} \\cdot \\text{{Shift Factor}}$")
        st.write(f"Calculation: $d_{{v50, adjusted}} = {results['dv50_original_fps']:.6f} \\text{{ ft}} \\cdot {results['shift_factor']:.3f} = {results['dv50_adjusted_fps']:.6f} \\text{{ ft}}$")
        st.success(f"**Result:** Adjusted Volume Median Diameter ($d_{{v50}}$) = **{results['dv50_adjusted_fps'] * FT_TO_MICRON:.2f} {micron_unit_label}** ({dv50_adjusted_display:.6f} {len_unit})")

        st.markdown("---")

        # Step 5: Calculate parameters for Upper-Limit Log Normal Distribution
        st.markdown("#### Step 5: Calculate Parameters for Upper-Limit Log Normal Distribution")
        d_max_display = from_fps(results['d_max_fps'], "length", unit_system)
        st.write(f"Using typical values from the article: $a = {A_DISTRIBUTION}$ and $\\delta = {DELTA_DISTRIBUTION}$.")
        st.write(f"Equation: $d_{{max}} = a \\cdot d_{{v50, adjusted}}$")
        st.write(f"Calculation: $d_{{max}} = {A_DISTRIBUTION} \\cdot {results['dv50_adjusted_fps']:.6f} \\text{{ ft}} = {results['d_max_fps']:.6f} \\text{{ ft}}$")
        st.success(f"**Result:** Maximum Droplet Size ($d_{{max}}$) = **{results['d_max_fps'] * FT_TO_MICRON:.2f} {micron_unit_label}** ({d_max_display:.6f} {len_unit})")

        st.markdown("---")

        st.info("Step 6 (Generating Droplet Size Distribution Data) is performed internally to prepare data for the plot.")

    else:
        st.warning("Please go to the 'Input Parameters' page and click 'Calculate Particle Size Distribution' first.")

# --- Page: Droplet Distribution Results ---
elif page == "Droplet Distribution Results":
    st.header("3. Particle Size Distribution Plot")

    if st.session_state.plot_data:
        plot_data = st.session_state.plot_data
        unit_system = st.session_state.unit_system

        # Define unit labels for plotting
        micron_unit_label = "microns" if unit_system == "FPS" else "µm"

        dp_values_microns = plot_data['dp_values_ft'] * FT_TO_MICRON
        volume_fraction = plot_data['volume_fraction']
        cumulative_volume_undersize = plot_data['cumulative_volume_undersize']
        cumulative_volume_oversize = [1 - v for v in cumulative_volume_undersize]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(dp_values_microns, cumulative_volume_undersize, 'o-', label='Cumulative Volume Undersize', markersize=2, color='#1f77b4')
        ax1.plot(dp_values_microns, cumulative_volume_oversize, 'o-', label='Cumulative Volume Oversize', markersize=2, color='#d62728')
        ax1.set_xlabel(f'Droplet Size ({micron_unit_label})', fontsize=12)
        ax1.set_ylabel('Cumulative Volume Fraction', color='black', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_ylim(0, 1.4)
        ax1.set_xlim(0, max(dp_values_microns) * 1.1 if dp_values_microns.size > 0 else 1000)

        ax2 = ax1.twinx()
        ax2.plot(dp_values_microns, volume_fraction, 'o-', label='Volume Fraction', markersize=2, color='#2ca02c')
        ax2.set_ylabel('Volume Fraction', color='black', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='black')
        
        max_fv = max(volume_fraction) if volume_fraction else 0.1
        ax2.set_ylim(0, max_fv * 1.2)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=10)

        plt.title('Entrainment Droplet Size Distribution', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    else:
        st.warning("Please go to the 'Input Parameters' page and click 'Calculate Particle Size Distribution' first to generate the plot data.")

st.markdown("""
---
#### Important Notes:
* **Figure 9 Approximation:** The "Droplet Size Distribution Shift Factor" is based on a simplified interpretation of Figure 9 from the article. For highly precise engineering applications, the curves in Figure 9 would need to be digitized and accurately modeled.
* **Log Normal Distribution Parameters:** The article states typical values for $a=4.0$ and $\delta=0.72$. The formula for $\delta$ shown in the article ($\\delta=\\frac{0.394}{log(\\frac{V_{so}}{V_{so}})}$) appears to be a typographical error, so the constant value $\\delta=0.72$ is used as indicated in the text.
* **Units:** The application now supports both Foot-Pound-Second (FPS) and International System (SI) units. All internal calculations are performed in FPS units to align with the article's correlations, with automatic conversions for inputs and outputs.
""")
