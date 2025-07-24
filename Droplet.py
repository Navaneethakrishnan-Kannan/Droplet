import streamlit as st
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from fpdf import FPDF # Import FPDF for PDF generation
import io # For handling in-memory image data

# --- Constants and Look-up Data (derived from the article) ---

# Table 1: Typical Liquid Surface Tension Values (dyne/cm)
# These will be converted to N/m for SI input/display
SURFACE_TENSION_TABLE_DYNE_CM = {
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
# Conversion: 1 dyne/cm = 0.001 N/m
DYNE_CM_TO_NM = 0.001

# Typical values for Upper-Limit Log Normal Distribution (from the article)
A_DISTRIBUTION = 4.0
DELTA_DISTRIBUTION = 0.72

# Figure 9 Approximation for Droplet Size Distribution Shift Factor
def get_shift_factor(inlet_device, rho_v_squared):
    """
    Approximates the droplet size distribution shift factor based on the inlet device
    and inlet momentum (rho_g * V_g^2) from Figure 9.
    The values are interpolated or set to a low value if beyond effective range.
    """
    if inlet_device == "No inlet device":
        return 1.0 # No shift if no inlet device
    elif inlet_device == "Diverter plate":
        # Curve starts at ~1.0, drops to ~0.6 at 2000, and near 0 by 2500
        if rho_v_squared <= 1000:
            return 1.0 - (0.1 * rho_v_squared / 1000)
        elif rho_v_squared <= 2000:
            # From (1000, 0.9) to (2000, 0.6)
            return 0.9 - (0.3 * (rho_v_squared - 1000) / 1000)
        elif rho_v_squared <= 2500:
            # From (2000, 0.6) to (2500, ~0.05)
            return 0.6 - (0.55 * (rho_v_squared - 2000) / 500)
        else: # Beyond effective capacity, very low shift factor
            return 0.05
    elif inlet_device == "Half-pipe":
        # Curve starts at ~1.0, drops sharply after ~1500, near 0.2 at 2000, and near 0 by 2500
        if rho_v_squared <= 1000:
            return 1.0 - (0.1 * rho_v_squared / 1000)
        elif rho_v_squared <= 1500:
            # From (1000, 0.9) to (1500, 0.7)
            return 0.9 - (0.2 * (rho_v_squared - 1000) / 500)
        elif rho_v_squared <= 2000:
            # From (1500, 0.7) to (2000, 0.2)
            return 0.7 - (0.5 * (rho_v_squared - 1500) / 500)
        elif rho_v_squared <= 2500:
            # From (2000, 0.2) to (2500, ~0.05)
            return 0.2 - (0.15 * (rho_v_squared - 2000) / 500)
        else: # Beyond effective capacity, very low shift factor as per user's example
            return 0.05
    elif inlet_device == "Vane-type":
        # Curve starts at ~1.0, drops slowly, around 0.8 at 6000, 0.5 at 10000
        if rho_v_squared <= 5000:
            return 1.0 - (0.05 * rho_v_squared / 5000) # Roughly 0.95 at 5000
        elif rho_v_squared <= 7000:
            # From (5000, ~0.95) to (7000, ~0.7)
            # To get 0.8 at 6000: (0.95 - 0.7) / 2 = 0.125. 0.95 - 0.125 = 0.825
            # Let's adjust points to hit 0.8 at 6000 more precisely if needed, or keep approximation
            # For 6000, (0.95 - (0.25 * (6000-5000)/2000)) = 0.95 - 0.125 = 0.825 (closer to 0.8)
            return 0.95 - (0.25 * (rho_v_squared - 5000) / 2000)
        elif rho_v_squared <= 10000:
            # From (7000, ~0.7) to (10000, ~0.5)
            return 0.7 - (0.2 * (rho_v_squared - 7000) / 3000)
        else:
            return 0.5 # Stays relatively high even at higher momentum
    elif inlet_device == "Cyclonic":
        # Curve starts at ~1.0, very slowly drops, around 0.95 at 6000, 0.9 at 10000, 0.2 at 14000
        if rho_v_squared <= 10000:
            # To get 0.95 at 6000: (1.0 - (0.1 * 6000 / 10000)) = 0.94. This is close to 0.95.
            return 1.0 - (0.1 * rho_v_squared / 10000)
        elif rho_v_squared <= 14000:
            return 0.9 - (0.7 * (rho_v_squared - 10000) / 4000)
        else:
            return 0.2 # Placeholder for very high values
    return 1.0 # Default if an unknown inlet device is somehow selected

# --- Unit Conversion Factors (for internal FPS calculation) ---
M_TO_FT = 3.28084 # 1 meter = 3.28084 feet
KG_TO_LB = 2.20462 # 1 kg = 2.20462 lb
MPS_TO_FTPS = 3.28084 # 1 m/s = 3.28084 ft/s
PAS_TO_LB_FT_S = 0.67197 # 1 Pa.s (kg/m.s) = 0.67197 lb/ft.s
KG_M3_TO_LB_FT3 = 0.0624279 # 1 kg/m^3 = 0.0624279 lb/ft^3
NM_TO_POUNDAL_FT = 2.2 # 1 N/m = 1000 dyne/cm; 1 dyne/cm = 0.0022 poundal/ft => 1 N/m = 2.2 poundal/ft

MICRON_TO_FT = 1e-6 * M_TO_FT
FT_TO_MICRON = 1 / MICRON_TO_FT

def to_fps(value, unit_type):
    """Converts a value from SI to FPS units for internal calculation."""
    if unit_type == "length": # meters to feet
        return value * M_TO_FT
    elif unit_type == "velocity": # m/s to ft/s
        return value * MPS_TO_FTPS
    elif unit_type == "density": # kg/m^3 to lb/ft^3
        return value * KG_M3_TO_LB_FT3
    elif unit_type == "viscosity": # Pa.s to lb/ft.s
        return value * PAS_TO_LB_FT_S
    elif unit_type == "surface_tension": # N/m to poundal/ft
        return value * NM_TO_POUNDAL_FT
    return value

def from_fps(value, unit_type):
    """Converts a value from FPS to SI units for display."""
    if unit_type == "length": # feet to meters
        return value / M_TO_FT
    elif unit_type == "velocity": # ft/s to m/s
        return value / MPS_TO_FTPS
    elif unit_type == "density": # lb/ft^3 to kg/m^3
        return value / KG_M3_TO_LB_FT3
    elif unit_type == "viscosity": # lb/ft.s to Pa.s
        return value / PAS_TO_LB_FT_S
    elif unit_type == "surface_tension": # poundal/ft to N/m
        return value / NM_TO_POUNDAL_FT
    elif unit_type == "momentum": # lb/ft-s^2 to Pa
        return value * 1.48816 # 1 lb/ft-s^2 = 1.48816 Pa
    return value

# --- PDF Report Generation Function ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Oil and Gas Separation: Particle Size Distribution Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, body)
        self.ln()

def generate_pdf_report(inputs, results, plot_image_buffer):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title Page
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 20, 'Oil and Gas Separation: Particle Size Distribution Analysis', 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Date: {st.session_state.report_date}', 0, 1, 'C')
    pdf.ln(20)

    # --- Input Parameters ---
    pdf.add_page()
    pdf.chapter_title('1. Input Parameters (SI Units)')
    # Replace problematic characters for PDF output
    pdf.chapter_body(f"Pipe Inside Diameter (D): {inputs['D_input']:.4f} m")
    pdf.chapter_body(f"Liquid Density (rho_l): {inputs['rho_l_input']:.2f} kg/m^3")
    pdf.chapter_body(f"Liquid Viscosity (mu_l): {inputs['mu_l_input']:.8f} Pa.s")
    pdf.chapter_body(f"Gas Velocity (Vg): {inputs['V_g_input']:.2f} m/s")
    pdf.chapter_body(f"Gas Density (rho_g): {inputs['rho_g_input']:.5f} kg/m^3")
    pdf.chapter_body(f"Gas Viscosity (mu_g): {inputs['mu_g_input']:.9f} Pa.s")
    
    sigma_display_val = from_fps(inputs['sigma_fps'], "surface_tension")
    pdf.chapter_body(f"Liquid Surface Tension (sigma): {sigma_display_val:.3f} N/m")
    pdf.chapter_body(f"Selected Inlet Device: {inputs['inlet_device']}")
    pdf.ln(5)

    # --- Calculation Steps ---
    pdf.add_page()
    pdf.chapter_title('2. Step-by-Step Calculation Results')
    
    # Define unit labels for SI system for report (using ASCII-safe versions)
    len_unit_pdf = "m"
    dens_unit_pdf = "kg/m^3"
    vel_unit_pdf = "m/s"
    visc_unit_pdf = "Pa.s"
    momentum_unit_pdf = "Pa"
    micron_unit_label_pdf = "um"

    pdf.set_font('Arial', 'B', 10)
    pdf.chapter_body("Inputs Used for Calculation (Converted to FPS for internal calculation):")
    pdf.set_font('Arial', '', 10)
    pdf.chapter_body(f"  Pipe Inside Diameter (D): {to_fps(inputs['D_input'], 'length'):.2f} ft")
    pdf.chapter_body(f"  Liquid Density (rho_l): {to_fps(inputs['rho_l_input'], 'density'):.2f} lb/ft^3")
    pdf.chapter_body(f"  Liquid Viscosity (mu_l): {to_fps(inputs['mu_l_input'], 'viscosity'):.7f} lb/ft-sec")
    pdf.chapter_body(f"  Gas Velocity (Vg): {to_fps(inputs['V_g_input'], 'velocity'):.2f} ft/sec")
    pdf.chapter_body(f"  Gas Density (rho_g): {to_fps(inputs['rho_g_input'], 'density'):.4f} lb/ft^3")
    pdf.chapter_body(f"  Gas Viscosity (mu_g): {to_fps(inputs['mu_g_input'], 'viscosity'):.8f} lb/ft-sec")
    pdf.chapter_body(f"  Liquid Surface Tension (sigma): {inputs['sigma_fps']:.4f} poundal/ft")
    pdf.chapter_body(f"  Selected Inlet Device: {inputs['inlet_device']}")
    pdf.ln(5)

    # Step 1
    pdf.set_font('Arial', 'B', 10)
    pdf.chapter_body("Step 1: Calculate Superficial Gas Reynolds Number (Re_g)")
    pdf.set_font('Arial', '', 10)
    pdf.chapter_body(f"Equation: Re_g = (D * V_g * rho_g) / mu_g")
    pdf.chapter_body(f"Calculation (FPS): Re_g = ({to_fps(inputs['D_input'], 'length'):.2f} ft * {to_fps(inputs['V_g_input'], 'velocity'):.2f} ft/sec * {to_fps(inputs['rho_g_input'], 'density'):.4f} lb/ft^3) / {to_fps(inputs['mu_g_input'], 'viscosity'):.8f} lb/ft-sec = {results['Re_g']:.2f}")
    pdf.chapter_body(f"Result: Superficial Gas Reynolds Number (Re_g) = {results['Re_g']:.2f} (dimensionless)")
    pdf.ln(5)

    # Step 2
    pdf.set_font('Arial', 'B', 10)
    pdf.chapter_body("Step 2: Calculate Initial Volume Median Diameter (d_v50) (Kataoka et al., 1983)")
    pdf.set_font('Arial', '', 10)
    pdf.chapter_body(f"Equation: d_v50 = 0.01 * (sigma / (rho_g V_g^2)) * Re_g^(2/3) * (rho_g / rho_l)^(-1/3) * (mu_g / mu_l)^(2/3)")
    pdf.chapter_body(f"Calculation (FPS): d_v50 = 0.01 * ({inputs['sigma_fps']:.4f} / ({to_fps(inputs['rho_g_input'], 'density'):.4f} * {to_fps(inputs['V_g_input'], 'velocity'):.2f}^2)) * ({results['Re_g']:.2f})^(2/3) * ({to_fps(inputs['rho_g_input'], 'density'):.4f} / {to_fps(inputs['rho_l_input'], 'density'):.2f})^(-1/3) * ({to_fps(inputs['mu_g_input'], 'viscosity'):.8f} / {to_fps(inputs['mu_l_input'], 'viscosity'):.7f})^(2/3) = {results['dv50_original_fps']:.6f} ft")
    pdf.chapter_body(f"Result: Initial Volume Median Diameter (d_v50) = {results['dv50_original_fps'] * FT_TO_MICRON:.2f} {micron_unit_label_pdf} ({from_fps(results['dv50_original_fps'], 'length'):.6f} {len_unit_pdf})")
    pdf.ln(5)

    # Step 3
    pdf.set_font('Arial', 'B', 10)
    pdf.chapter_body("Step 3: Calculate Inlet Momentum (rho_g V_g^2)")
    pdf.set_font('Arial', '', 10)
    pdf.chapter_body(f"Equation: rho_g V_g^2 = rho_g * V_g^2")
    pdf.chapter_body(f"Calculation (FPS): rho_g V_g^2 = {to_fps(inputs['rho_g_input'], 'density'):.4f} lb/ft^3 * ({to_fps(inputs['V_g_input'], 'velocity'):.2f} ft/sec)^2 = {results['rho_v_squared_fps']:.2f} lb/ft-sec^2")
    pdf.chapter_body(f"Result: Inlet Momentum (rho_g V_g^2) = {from_fps(results['rho_v_squared_fps'], 'momentum'):.2f} {momentum_unit_pdf}")
    pdf.ln(5)

    # Step 4
    pdf.set_font('Arial', 'B', 10)
    pdf.chapter_body("Step 4: Apply Inlet Device 'Droplet Size Distribution Shift Factor'")
    pdf.set_font('Arial', '', 10)
    pdf.chapter_body(f"Selected Inlet Device: {inputs['inlet_device']}")
    pdf.chapter_body(f"Estimated Shift Factor (from Fig. 9): {results['shift_factor']:.3f}")
    pdf.chapter_body(f"Equation: d_v50,adjusted = d_v50,original * Shift Factor")
    pdf.chapter_body(f"Calculation (FPS): d_v50,adjusted = {results['dv50_original_fps']:.6f} ft * {results['shift_factor']:.3f} = {results['dv50_adjusted_fps']:.6f} ft")
    pdf.chapter_body(f"Result: Adjusted Volume Median Diameter (d_v50) = {results['dv50_adjusted_fps'] * FT_TO_MICRON:.2f} {micron_unit_label_pdf} ({from_fps(results['dv50_adjusted_fps'], 'length'):.6f} {len_unit_pdf})")
    pdf.ln(5)

    # Step 5
    pdf.set_font('Arial', 'B', 10)
    pdf.chapter_body("Step 5: Calculate Parameters for Upper-Limit Log Normal Distribution")
    pdf.set_font('Arial', '', 10)
    pdf.chapter_body(f"Using typical values from the article: a = {A_DISTRIBUTION} and delta = {DELTA_DISTRIBUTION}.")
    pdf.chapter_body(f"Equation: d_max = a * d_v50,adjusted")
    pdf.chapter_body(f"Calculation (FPS): d_max = {A_DISTRIBUTION} * {results['dv50_adjusted_fps']:.6f} ft = {results['d_max_fps']:.6f} ft")
    pdf.chapter_body(f"Result: Maximum Droplet Size (d_max) = {results['d_max_fps'] * FT_TO_MICRON:.2f} {micron_unit_label_pdf} ({from_fps(results['d_max_fps'], 'length'):.6f} {len_unit_pdf})")
    pdf.ln(5)

    # --- Droplet Distribution Plot ---
    pdf.add_page()
    pdf.chapter_title('3. Droplet Distribution Results')
    pdf.chapter_body("The following graph shows the calculated entrainment droplet size distribution:")
    
    # Add the plot image
    if plot_image_buffer:
        pdf.image(plot_image_buffer, x=10, y=pdf.get_y(), w=pdf.w - 20)
    pdf.ln(5)

    return pdf.output(dest='S').encode('latin1') # Return PDF as bytes

# --- Streamlit App Layout ---

st.set_page_config(layout="centered", page_title="Oil & Gas Separation App")

st.title("🛢️ Oil and Gas Separation: Particle Size Distribution")
st.markdown("""
This application helps quantify the entrained liquid droplet size distribution in gas-liquid separators,
based on the principles and correlations discussed in the article "Quantifying Separation Performance" by Mark Bothamley.
All inputs and outputs are in **SI Units**.
""")

# Initialize session state for inputs and results if not already present
if 'inputs' not in st.session_state:
    # Initialize inputs with default SI values
    st.session_state.inputs = {
        'D_input': 0.3048, # m (1 ft)
        'rho_l_input': 640.7, # kg/m3 (40 lb/ft3)
        'mu_l_input': 0.000743, # Pa.s (0.0005 lb/ft-sec)
        'V_g_input': 15.24, # m/s (50 ft/sec)
        'rho_g_input': 1.6018, # kg/m3 (0.1 lb/ft3)
        'mu_g_input': 0.00001488, # Pa.s (0.00001 lb/ft-sec)
        'surface_tension_option': "Water/gas",
        'sigma_custom': 0.03, # N/m (30 dyne/cm)
        'inlet_device': "No inlet device",
    }
    # Initialize sigma_fps based on the initial default surface tension option
    st.session_state.inputs['sigma_fps'] = SURFACE_TENSION_TABLE_DYNE_CM["Water/gas"] * DYNE_CM_TO_POUNDAL_FT

if 'calculation_results' not in st.session_state:
    st.session_state.calculation_results = None
if 'plot_data' not in st.session_state:
    st.session_state.plot_data = None
if 'report_date' not in st.session_state:
    st.session_state.report_date = ""

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Input Parameters", "Calculation Steps", "Droplet Distribution Results"])

# --- Page: Input Parameters ---
if page == "Input Parameters":
    st.header("1. Input Parameters (SI Units)")

    # Define unit labels for SI system
    len_unit = "m"
    dens_unit = "kg/m³"
    vel_unit = "m/s"
    visc_unit = "Pa·s"
    surf_tens_input_unit = "N/m"

    st.subheader("Feed Pipe Conditions")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.inputs['D_input'] = st.number_input(f"Pipe Inside Diameter ({len_unit})", min_value=0.001, value=st.session_state.inputs['D_input'], format="%.4f", key='D_input_widget',
                            help="Diameter of the feed pipe to the separator.")
        st.session_state.inputs['rho_l_input'] = st.number_input(f"Liquid Density ({dens_unit})", min_value=0.1, value=st.session_state.inputs['rho_l_input'], format="%.2f", key='rho_l_input_widget',
                                help="Density of the liquid phase.")
        st.session_state.inputs['mu_l_input'] = st.number_input(f"Liquid Viscosity ({visc_unit})", min_value=1e-8, value=st.session_state.inputs['mu_l_input'], format="%.8f", key='mu_l_input_widget',
                                help=f"Viscosity of the liquid phase. Example: Water at 20°C is ~0.001 Pa·s.")
    with col2:
        st.session_state.inputs['V_g_input'] = st.number_input(f"Gas Velocity ({vel_unit})", min_value=0.01, value=st.session_state.inputs['V_g_input'], format="%.2f", key='V_g_input_widget',
                              help="Superficial gas velocity in the feed pipe.")
        st.session_state.inputs['rho_g_input'] = st.number_input(f"Gas Density ({dens_unit})", min_value=1e-5, value=st.session_state.inputs['rho_g_input'], format="%.5f", key='rho_g_input_widget',
                                help="Density of the gas phase.")
        st.session_state.inputs['mu_g_input'] = st.number_input(f"Gas Viscosity ({visc_unit})", min_value=1e-9, value=st.session_state.inputs['mu_g_input'], format="%.9f", key='mu_g_input_widget',
                                help=f"Viscosity of the gas phase. Example: Methane at 20°C is ~0.000011 Pa·s.")

    st.markdown("---")
    col_st, col_id = st.columns(2)

    with col_st:
        st.subheader("Liquid Surface Tension")
        # Convert SURFACE_TENSION_TABLE_DYNE_CM values to N/m for display in selectbox
        st_options_nm = {k: v * DYNE_CM_TO_NM for k, v in SURFACE_TENSION_TABLE_DYNE_CM.items()}
        
        # Determine current index for selectbox
        current_st_option_index = list(SURFACE_TENSION_TABLE_DYNE_CM.keys()).index(st.session_state.inputs['surface_tension_option']) if st.session_state.inputs['surface_tension_option'] in SURFACE_TENSION_TABLE_DYNE_CM else len(SURFACE_TENSION_TABLE_DYNE_CM.keys())
        
        st.session_state.inputs['surface_tension_option'] = st.selectbox(
            "Select Fluid System for Surface Tension",
            options=list(SURFACE_TENSION_TABLE_DYNE_CM.keys()) + ["Custom"],
            index=current_st_option_index,
            key='surface_tension_option_select',
            format_func=lambda x: f"{x} ({st_options_nm[x]:.3f} N/m)" if x in st_options_nm else x, # Show N/m in options
            help="Choose a typical value or enter a custom one."
        )

        sigma_input_val_si = 0.0
        if st.session_state.inputs['surface_tension_option'] == "Custom":
            st.session_state.inputs['sigma_custom'] = st.number_input(f"Custom Liquid Surface Tension ({surf_tens_input_unit})", min_value=0.0001, value=st.session_state.inputs['sigma_custom'], format="%.4f", key='sigma_custom_input')
            sigma_input_val_si = st.session_state.inputs['sigma_custom']
        else:
            sigma_input_val_si = SURFACE_TENSION_TABLE_DYNE_CM[st.session_state.inputs['surface_tension_option']] * DYNE_CM_TO_NM
        
        # Store sigma in FPS for internal calculation
        st.session_state.inputs['sigma_fps'] = to_fps(sigma_input_val_si, "surface_tension")
        
        st.info(f"**Selected Liquid Surface Tension:** {sigma_input_val_si:.3f} {surf_tens_input_unit}")


    with col_id:
        st.subheader("Separator Inlet Device")
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
        import datetime
        st.session_state.report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # Convert all SI inputs to FPS for consistent calculation
            D = to_fps(st.session_state.inputs['D_input'], "length")
            rho_l = to_fps(st.session_state.inputs['rho_l_input'], "density")
            mu_l = to_fps(st.session_state.inputs['mu_l_input'], "viscosity")
            V_g = to_fps(st.session_state.inputs['V_g_input'], "velocity")
            rho_g = to_fps(st.session_state.inputs['rho_g_input'], "density")
            mu_g = to_fps(st.session_state.inputs['mu_g_input'], "viscosity")
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
            
            # Corrected line: Changed Re_g**2 to Re_g**(2/3) as per user request
            dv50_original_fps = 0.01 * (sigma / (rho_g * V_g**2)) * (Re_g**(2/3)) * ((rho_g / rho_l)**(-1/3)) * ((mu_g / mu_l)**(2/3))
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
                # Ensure dp is not equal to d_max_fps to avoid division by zero in log argument
                if dp >= d_max_fps:
                    z_val = np.inf # Or handle as per distribution definition for upper limit
                else:
                    z_val = np.log((A_DISTRIBUTION * dp) / (d_max_fps - dp))
                
                # Handle potential division by zero for fv_dp if dp or (d_max_fps - dp) is zero
                if dp == 0 or (d_max_fps - dp) == 0:
                    fv_dp = 0
                else:
                    fv_dp = ((DELTA_DISTRIBUTION * d_max_fps) / (np.sqrt(np.pi * dp * (d_max_fps - dp)))) * np.exp(-DELTA_DISTRIBUTION**2 * z_val**2)
                
                volume_fraction.append(fv_dp)
                
                # For cumulative, erf(inf) is 1, erf(-inf) is -1
                if z_val == np.inf:
                    v_under = 1.0
                elif z_val == -np.inf:
                    v_under = 0.0
                else:
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
        
        # Define unit labels for SI system
        len_unit = "m"
        dens_unit = "kg/m³"
        vel_unit = "m/s"
        visc_unit = "Pa·s"
        momentum_unit = "Pa"
        micron_unit_label = "µm"

        # Display inputs used for calculation (original SI values)
        st.subheader("Inputs Used for Calculation (SI Units)")
        st.write(f"Pipe Inside Diameter (D): {st.session_state.inputs['D_input']:.4f} {len_unit}")
        st.write(f"Liquid Density (ρl): {st.session_state.inputs['rho_l_input']:.2f} {dens_unit}")
        st.write(f"Liquid Viscosity (μl): {st.session_state.inputs['mu_l_input']:.8f} {visc_unit}")
        st.write(f"Gas Velocity (Vg): {st.session_state.inputs['V_g_input']:.2f} {vel_unit}")
        st.write(f"Gas Density (ρg): {st.session_state.inputs['rho_g_input']:.5f} {dens_unit}")
        st.write(f"Gas Viscosity (μg): {st.session_state.inputs['mu_g_input']:.9f} {visc_unit}")
        # Display selected surface tension in SI units
        sigma_display_val = from_fps(st.session_state.inputs['sigma_fps'], "surface_tension")
        st.write(f"Liquid Surface Tension (σ): {sigma_display_val:.3f} N/m")
        st.write(f"Selected Inlet Device: {st.session_state.inputs['inlet_device']}")
        st.markdown("---")

        # Step 1: Calculate Superficial Gas Reynolds Number (Re_g)
        st.markdown("#### Step 1: Calculate Superficial Gas Reynolds Number ($Re_g$)")
        D_fps = to_fps(st.session_state.inputs['D_input'], "length")
        V_g_fps = to_fps(st.session_state.inputs['V_g_input'], "velocity")
        rho_g_fps = to_fps(st.session_state.inputs['rho_g_input'], "density")
        mu_g_fps = to_fps(st.session_state.inputs['mu_g_input'], "viscosity")

        st.write(f"Equation: $Re_g = \\frac{{D \\cdot V_g \\cdot \\rho_g}}{{\\mu_g}}$")
        st.write(f"Calculation (FPS): $Re_g = \\frac{{{D_fps:.2f} \\text{{ ft}} \\cdot {V_g_fps:.2f} \\text{{ ft/sec}} \\cdot {rho_g_fps:.4f} \\text{{ lb/ft}}^3}}{{{mu_g_fps:.8f} \\text{{ lb/ft-sec}}}} = {results['Re_g']:.2f}$")
        st.success(f"**Result:** Superficial Gas Reynolds Number ($Re_g$) = **{results['Re_g']:.2f}** (dimensionless)")

        st.markdown("---")

        # Step 2: Calculate Volume Median Diameter ($d_{v50}$) without inlet device effect
        st.markdown("#### Step 2: Calculate Initial Volume Median Diameter ($d_{v50}$) (Kataoka et al., 1983)")
        rho_l_fps = to_fps(st.session_state.inputs['rho_l_input'], "density")
        mu_l_fps = to_fps(st.session_state.inputs['mu_l_input'], "viscosity")

        dv50_original_display = from_fps(results['dv50_original_fps'], "length")
        
        # Updated LaTeX formula for display
        st.write(f"Equation: $d_{{v50}} = 0.01 \\left(\\frac{{\\sigma}}{{\\rho_g V_g^2}}\\right) Re_g^{{2/3}} \\left(\\frac{{\\rho_g}}{{\\rho_l}}\\right)^{{-1/3}} \\left(\\frac{{\\mu_g}}{{\\mu_l}}\\right)^{{2/3}}$")
        st.write(f"Calculation (FPS): $d_{{v50}} = 0.01 \\left(\\frac{{{st.session_state.inputs['sigma_fps']:.4f}}}{{{rho_g_fps:.4f} \\cdot ({V_g_fps:.2f})^2}}\\right) ({results['Re_g']:.2f})^{{2/3}} \\left(\\frac{{{rho_g_fps:.4f}}}{{{rho_l_fps:.2f}}}\\right)^{{-0.333}} \\left(\\frac{{{mu_g_fps:.8f}}}{{{mu_l_fps:.7f}}}\\right)^{{0.667}}$")
        st.success(f"**Result:** Initial Volume Median Diameter ($d_{{v50}}$) = **{results['dv50_original_fps'] * FT_TO_MICRON:.2f} {micron_unit_label}** ({dv50_original_display:.6f} {len_unit})")

        st.markdown("---")

        # Step 3: Determine Inlet Momentum (rho_g V_g^2)
        st.markdown("#### Step 3: Calculate Inlet Momentum ($\\rho_g V_g^2$)")
        rho_v_squared_display = from_fps(results['rho_v_squared_fps'], "momentum")
        st.write(f"Equation: $\\rho_g V_g^2 = \\rho_g \\cdot V_g^2$")
        st.write(f"Calculation (FPS): $\\rho_g V_g^2 = {rho_g_fps:.4f} \\text{{ lb/ft}}^3 \\cdot ({V_g_fps:.2f} \\text{{ ft/sec}})^2 = {results['rho_v_squared_fps']:.2f} \\text{{ lb/ft-sec}}^2$")
        st.success(f"**Result:** Inlet Momentum ($\\rho_g V_g^2$) = **{rho_v_squared_display:.2f} {momentum_unit}**")

        st.markdown("---")

        # Step 4: Apply Inlet Device "Droplet Size Distribution Shift Factor"
        st.markdown("#### Step 4: Apply Inlet Device Effect (Droplet Size Distribution Shift Factor)")
        st.write(f"Selected Inlet Device: **{st.session_state.inputs['inlet_device']}**")
        dv50_adjusted_display = from_fps(results['dv50_adjusted_fps'], "length")
        st.write(f"Based on Figure 9 from the article, for an inlet momentum of {rho_v_squared_display:.2f} {momentum_unit} and a '{st.session_state.inputs['inlet_device']}' device, the estimated shift factor is **{results['shift_factor']:.3f}**.")
        st.write(f"Equation: $d_{{v50, adjusted}} = d_{{v50, original}} \\cdot \\text{{Shift Factor}}$")
        st.write(f"Calculation (FPS): $d_{{v50, adjusted}} = {results['dv50_original_fps']:.6f} \\text{{ ft}} \\cdot {results['shift_factor']:.3f} = {results['dv50_adjusted_fps']:.6f} \\text{{ ft}}$")
        st.success(f"**Result:** Adjusted Volume Median Diameter ($d_{{v50}}$) = **{results['dv50_adjusted_fps'] * FT_TO_MICRON:.2f} {micron_unit_label}** ({dv50_adjusted_display:.6f} {len_unit})")

        st.markdown("---")

        # Step 5: Calculate parameters for Upper-Limit Log Normal Distribution
        st.markdown("#### Step 5: Calculate Parameters for Upper-Limit Log Normal Distribution")
        d_max_display = from_fps(results['d_max_fps'], "length")
        st.write(f"Using typical values from the article: $a = {A_DISTRIBUTION}$ and $\\delta = {DELTA_DISTRIBUTION}$.")
        st.write(f"Equation: $d_{{max}} = a \\cdot d_{{v50, adjusted}}$")
        st.write(f"Calculation (FPS): $d_{{max}} = {A_DISTRIBUTION} \\cdot {results['dv50_adjusted_fps']:.6f} \\text{{ ft}} = {results['d_max_fps']:.6f} \\text{{ ft}}$")
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
        
        # Define unit labels for plotting
        micron_unit_label = "µm" # Always SI for this version

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

        # Save plot to a BytesIO object for PDF embedding
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        buf.seek(0) # Rewind to the beginning of the buffer

        st.download_button(
            label="Download Report as PDF",
            data=generate_pdf_report(st.session_state.inputs, st.session_state.calculation_results, buf),
            file_name="Droplet_Distribution_Report.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Please go to the 'Input Parameters' page and click 'Calculate Particle Size Distribution' first to generate the plot data.")

st.markdown("""
---
#### Important Notes:
* **Figure 9 Approximation:** The "Droplet Size Distribution Shift Factor" is based on a simplified interpretation of Figure 9 from the article. For highly precise engineering applications, the curves in Figure 9 would need to be digitized and accurately modeled.
* **Log Normal Distribution Parameters:** The article states typical values for $a=4.0$ and $\delta=0.72$. The formula for $\delta$ shown in the article ($\\delta=\\frac{0.394}{log(\\frac{V_{so}}{V_{so}})}$) appears to be a typographical error, so the constant value $\\delta=0.72$ is used as indicated in the text.
* **Units:** This application now exclusively uses the International System (SI) units for all inputs and outputs. All internal calculations are still performed in FPS units to align with the article's correlations, with automatic conversions handled internally.
""")
