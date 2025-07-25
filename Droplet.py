import streamlit as st
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from fpdf import FPDF # Import FPDF for PDF generation
import io # For handling in-memory image data
import pandas as pd # Import pandas for data table
import math # Import math for new calculations

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

# Figure 9 Approximation Data for Droplet Size Distribution Shift Factor
# Data extracted from the provided image (Figure 9 table)
SHIFT_FACTOR_DATA = {
    "No inlet device": {
        "rho_v_squared": np.array([0, 100, 200, 300, 400, 500, 600, 650, 675]),
        "shift_factor": np.array([1.00, 0.98, 0.95, 0.90, 0.77, 0.50, 0.20, 0.08, 0.08]) # Using 0.08 as per image
    },
    "Diverter plate": {
        "rho_v_squared": np.array([0,31, 148, 269, 368, 574, 626, 660, 744, 775, 812, 870, 903, 941, 952]),
        "shift_factor": np.array([1,0.99, 0.95, 0.91, 0.87, 0.79, 0.75, 0.71, 0.62, 0.56, 0.52, 0.48, 0.43, 0.35, 0.39])
    },
    "Half-pipe": {
        "rho_v_squared": np.array([0,287, 579, 843, 1064, 1236, 1389, 1519, 1587, 1659, 1743, 1782, 1819, 1892, 1917]),
        "shift_factor": np.array([1,0.96, 0.94, 0.90, 0.87, 0.83, 0.80, 0.71, 0.67, 0.57, 0.43, 0.35, 0.25, 0.07, 0.04])
    },
    "Vane-type": {
        "rho_v_squared": np.array([0,1433, 2297, 3162, 4026, 4891, 5323, 5754, 6229, 6583, 6686, 6775, 6862, 6891, 6894, 6979, 7070]),
        "shift_factor": np.array([1,0.99, 0.97, 0.95, 0.92, 0.89, 0.87, 0.83, 0.78, 0.72, 0.66, 0.60, 0.54, 0.48, 0.43, 0.37, 0.31])
    },
    "Cyclonic": {
        "rho_v_squared": np.array([0,553, 2294, 2716, 4014, 5312, 5745, 7043, 7908, 8340, 8772, 9205, 9637, 10069, 10501, 10932, 11364, 11794, 12169, 12467, 12716, 12923, 13123, 13323, 13509, 13688, 13891]),
        "shift_factor": np.array([1,0.99, 0.98, 0.97, 1.00, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.87, 0.86, 0.84, 0.81, 0.77, 0.72, 0.67, 0.61, 0.56, 0.50, 0.45, 0.39, 0.34, 0.28, 0.23, 0.18])
    }
}


def get_shift_factor(inlet_device, rho_v_squared):
    """
    Calculates the droplet size distribution shift factor using linear interpolation
    based on the inlet device and inlet momentum (rho_g * V_g^2) from Figure 9 data.
    Handles out-of-range values by clamping to the nearest boundary and provides a warning.
    """
    if inlet_device not in SHIFT_FACTOR_DATA:
        st.warning(f"Unknown inlet device: '{inlet_device}'. Defaulting shift factor to 1.0.")
        return 1.0

    data = SHIFT_FACTOR_DATA[inlet_device]
    x_values = data["rho_v_squared"]
    y_values = data["shift_factor"]

    # Check if rho_v_squared is outside the defined range
    if rho_v_squared < x_values.min():
        st.warning(f"Inlet momentum ({rho_v_squared:.2f} lb/ft-sec^2) is below the minimum defined for '{inlet_device}' ({x_values.min():.2f} lb/ft-sec^2). Using minimum shift factor: {y_values.min():.3f}.")
        return y_values.min() # Use the smallest shift factor if below range
    elif rho_v_squared > x_values.max():
        # Corrected: Use the minimum shift factor for values above the maximum defined range
        st.warning(f"Inlet momentum ({rho_v_squared:.2f} lb/ft-sec^2) is above the maximum defined for '{inlet_device}' ({x_values.max():.2f} lb/ft-sec^2). Using minimum shift factor: {y_values.min():.3f}.")
        return y_values.min() # Use the smallest shift factor if above range

    # Perform linear interpolation
    shift_factor = np.interp(rho_v_squared, x_values, y_values)
    return float(shift_factor)


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
    elif unit_type == "momentum": # lb/ft-s^2 to Pa
        return value * 1.48816 # 1 lb/ft-s^2 = 1.48816 Pa
    return value

# --- Functions for E (Entrainment Fraction) Calculation ---
def calculate_e_from_specific_equation(Ug_val, Wl_mass_flow):
    """
    Calculates E using one of the five specific empirical equations.
    
    Args:
        Ug_val (int/float): The specific Ug value (6, 7, 9, 11, or 34).
        Wl_mass_flow (float): The Total Liquid Mass Flow Rate (kg/s), used directly as Wl for empirical equation.
        
    Returns:
        float: The calculated E value.
        
    Raises:
        ValueError: If Ug_val is not one of the predefined values.
    """
    if Ug_val == 6:
        A = 0.35
        B = -0.07
        C = 3.65
    elif Ug_val == 7:
        A = 0.5
        B = -0.1
        C = 3.8
    elif Ug_val == 9:
        A = 0.74
        B = -0.15
        C = 3.67
    elif Ug_val == 11:
        A = 0.86
        B = -0.19
        C = 4.09
    elif Ug_val == 34:
        A = 0.99
        B = -0.19
        C = 3.85
    else:
        # This case should ideally not be reached if clamping is done correctly before calling.
        # It's a safeguard if this function is called directly with an unhandled Ug_val.
        raise ValueError(f"No specific equation found for Ug = {Ug_val}. Must be one of 6, 7, 9, 11, 34.")
    
    # Wl_mass_flow is directly used as Wl as per user's instruction
    return A + B * math.exp(-C * Wl_mass_flow)

def calculate_e_interpolated(Ug_target, Wl_mass_flow):
    """
    Calculates E for a given Ug_target and Wl_mass_flow using linear interpolation
    between the known specific Ug equations. Handles out-of-range Ug_target
    by clamping to the nearest boundary (6 or 34).
    
    Args:
        Ug_target (float): The target Ug value (gas velocity in m/s) for which to calculate E.
        Wl_mass_flow (float): The Total Liquid Mass Flow Rate (kg/s), used directly as Wl for empirical equation.
        
    Returns:
        float: The interpolated E value (entrainment fraction).
    """
    known_ug_values = [6, 7, 9, 11, 34]
    
    # Clamp Ug_target to the valid range
    clamped_ug_target = max(min(Ug_target, max(known_ug_values)), min(known_ug_values))
    
    if clamped_ug_target in known_ug_values:
        # If clamped_ug_target is one of the known values, use its direct equation
        return calculate_e_from_specific_equation(clamped_ug_target, Wl_mass_flow)

    # Find the two bracketing Ug values for the clamped target
    Ug_low = None
    Ug_high = None
    for i in range(len(known_ug_values) - 1):
        if known_ug_values[i] < clamped_ug_target < known_ug_values[i+1]:
            Ug_low = known_ug_values[i]
            Ug_high = known_ug_values[i+1]
            break
    
    if Ug_low is None or Ug_high is None:
        # This case should not be reached with the clamping logic, but as a safeguard.
        # This might happen if Ug_target is exactly equal to the max or min known_ug_values
        # but not caught by the `in known_ug_values` check due to float precision.
        # For robustness, we can return the clamped value's direct calculation here too.
        return calculate_e_from_specific_equation(clamped_ug_target, Wl_mass_flow)


    # Calculate E at the lower and higher Ug values using their specific equations
    E_low = calculate_e_from_specific_equation(Ug_low, Wl_mass_flow)
    E_high = calculate_e_from_specific_equation(Ug_high, Wl_mass_flow)
    
    # Perform linear interpolation
    E_interpolated = E_low + (E_high - E_low) * \
                     ((clamped_ug_target - Ug_low) / (Ug_high - Ug_low))
                     
    return E_interpolated


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
        
    def add_table(self, headers, data, col_widths):
        # Set font for table headers
        self.set_font('Arial', 'B', 9)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, 1, 0, 'C')
        self.ln()

        # Set font for table data
        self.set_font('Arial', '', 8)
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], 6, str(item), 1, 0, 'C')
            self.ln()
        self.ln(5)


def generate_pdf_report(inputs, results, plot_image_buffer, plot_data_for_table):
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
    pdf.chapter_body(f"Total Liquid Mass Flow Rate: {inputs['Q_liquid_mass_flow_rate_input']:.2f} kg/s") # New input
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
    mass_flow_unit_pdf = "kg/s"
    vol_flow_unit_pdf = "m^3/s" # New unit for PDF

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
    pdf.chapter_body(f"  Total Liquid Mass Flow Rate: {inputs['Q_liquid_mass_flow_rate_input']:.2f} {mass_flow_unit_pdf}") # New input
    pdf.chapter_body(f"  Number of Points for Distribution: {inputs['num_points_distribution']}") # New input
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

    # Step 6: Entrainment Fraction (E) Calculation
    pdf.set_font('Arial', 'B', 10)
    pdf.chapter_body("Step 6: Calculate Entrainment Fraction (E)")
    pdf.set_font('Arial', '', 10)
    pdf.chapter_body(f"Gas Velocity (Ug): {inputs['V_g_input']:.2f} m/s")
    pdf.chapter_body(f"Liquid Loading (Wl): {inputs['Q_liquid_mass_flow_rate_input']:.2f} {mass_flow_unit_pdf}")
    pdf.chapter_body(f"Result: Entrainment Fraction (E) = {results['E_fraction']:.4f} (dimensionless)")
    pdf.chapter_body(f"Result: Total Entrained Liquid Mass Flow Rate = {results['Q_entrained_total_mass_flow_rate_si']:.4f} {mass_flow_unit_pdf}")
    pdf.chapter_body(f"Result: Total Entrained Liquid Volume Flow Rate = {results['Q_entrained_total_volume_flow_rate_si']:.6f} {vol_flow_unit_pdf}") # New total volume flow
    pdf.ln(5)


    # --- Droplet Distribution Plot ---
    pdf.add_page()
    pdf.chapter_title('3. Droplet Distribution Results')
    pdf.chapter_body("The following graph shows the calculated entrainment droplet size distribution:")
    
    # Add the plot image
    if plot_image_buffer:
        pdf.image(plot_image_buffer, x=10, y=pdf.get_y(), w=pdf.w - 20)
    pdf.ln(5)

    # --- Volume Fraction Data Table ---
    pdf.add_page() # Start a new page for the table
    pdf.chapter_title('4. Volume Fraction Data Table (Sampled)')
    if plot_data_for_table and 'dp_values_microns' in plot_data_for_table and len(plot_data_for_table['dp_values_microns']) > 0:
        headers = ["Droplet Size (um)", "Volume Fraction", "Cumulative Undersize", "Entrained Mass Flow (kg/s)", "Entrained Volume Flow (m^3/s)"] # Removed Cumulative Oversize
        # Use all points for the PDF table
        full_data = []
        for i in range(len(plot_data_for_table['dp_values_microns'])):
            full_data.append([
                f"{plot_data_for_table['dp_values_microns'][i]:.2f}",
                f"{plot_data_for_table['volume_fraction'][i]:.4f}", # Use normalized for table
                f"{plot_data_for_table['cumulative_volume_undersize'][i]:.4f}",
                f"{plot_data_for_table['entrained_mass_flow_rate_per_dp'][i]:.6f}",
                f"{plot_data_for_table['entrained_volume_flow_rate_per_dp'][i]:.9f}" # Added Volume Flow
            ])
        
        col_widths = [25, 25, 35, 40, 45] # Adjusted column widths
        pdf.add_table(headers, full_data, col_widths)
    else:
        pdf.chapter_body("No data available to display in the table. Please check your input parameters.")


    return bytes(pdf.output(dest='S')) # Return PDF as bytes directly

# --- Streamlit App Layout ---

st.set_page_config(layout="centered", page_title="Oil & Gas Separation App")

st.title("🛢️ Oil and Gas Separation: Particle Size Distribution")
st.markdown("""
This application helps quantify the entrained liquid droplet size distribution in gas-liquid separators,
based on the principles and correlations discussed in the article "Quantifying Separation Performance" by Mark Bothamley.
All inputs and outputs are in **SI Units**.
""")

# --- Function to perform all calculations ---
def _calculate_all_data(inputs, num_points_override=None):
    """Performs all calculations and returns results and plot_data."""
    results = {}
    plot_data = {
        'dp_values_ft': [],
        'volume_fraction': [],
        'cumulative_volume_undersize': [],
        'cumulative_volume_oversize': [],
        'entrained_mass_flow_rate_per_dp': [],
        'entrained_volume_flow_rate_per_dp': []
    }

    # Convert all SI inputs to FPS for consistent calculation
    D = to_fps(inputs['D_input'], "length")
    rho_l = to_fps(inputs['rho_l_input'], "density")
    mu_l = to_fps(inputs['mu_l_input'], "viscosity")
    V_g = to_fps(inputs['V_g_input'], "velocity")
    rho_g = to_fps(inputs['rho_g_input'], "density")
    mu_g = to_fps(inputs['mu_g_input'], "viscosity")
    sigma = inputs['sigma_fps'] # This is already in poundal/ft

    # Step 1: Calculate Superficial Gas Reynolds Number (Re_g)
    if mu_g == 0: raise ValueError("Gas viscosity (μg) cannot be zero for Reynolds number calculation.")
    Re_g = (D * V_g * rho_g) / mu_g
    results['Re_g'] = Re_g

    # Step 2: Calculate Volume Median Diameter ($d_{v50}$) without inlet device effect
    if V_g == 0 or rho_g == 0 or rho_l == 0 or mu_l == 0:
        raise ValueError("Gas velocity, gas density, liquid density, and liquid viscosity must be non-zero for $d_{v50}$ calculation.")
    
    dv50_original_fps = 0.01 * (sigma / (rho_g * V_g**2)) * (Re_g**(2/3)) * ((rho_g / rho_l)**(-1/3)) * ((mu_g / mu_l)**(2/3))
    results['dv50_original_fps'] = dv50_original_fps

    # Step 3: Determine Inlet Momentum (rho_g V_g^2)
    rho_v_squared_fps = rho_g * V_g**2
    results['rho_v_squared_fps'] = rho_v_squared_fps

    # Step 4: Apply Inlet Device "Droplet Size Distribution Shift Factor"
    shift_factor = get_shift_factor(inputs['inlet_device'], rho_v_squared_fps)
    dv50_adjusted_fps = dv50_original_fps * shift_factor
    results['shift_factor'] = shift_factor
    results['dv50_adjusted_fps'] = dv50_adjusted_fps

    # Step 5: Calculate parameters for Upper-Limit Log Normal Distribution
    d_max_fps = A_DISTRIBUTION * dv50_adjusted_fps
    results['d_max_fps'] = d_max_fps

    # Step 6: Generate Volume Fraction (PDF values) and Cumulative Volume Fraction for a range of droplet sizes
    dp_min_calc_fps = dv50_adjusted_fps * 0.01
    dp_max_calc_fps = d_max_fps * 0.999

    # Use the override if provided, otherwise use the value from inputs
    num_points = num_points_override if num_points_override is not None else inputs['num_points_distribution']
    dp_values_ft = np.linspace(dp_min_calc_fps, dp_max_calc_fps, num_points)
    
    volume_fraction_pdf_values = []

    for dp in dp_values_ft:
        if dp >= d_max_fps:
            z_val = np.inf
        else:
            z_val = np.log((A_DISTRIBUTION * dp) / (d_max_fps - dp))
        
        if dp == 0 or (d_max_fps - dp) == 0:
            fv_dp = 0
        else:
            fv_dp = ((DELTA_DISTRIBUTION * d_max_fps) / (np.sqrt(np.pi * dp * (d_max_fps - dp)))) * np.exp(-DELTA_DISTRIBUTION**2 * z_val**2)
        
        volume_fraction_pdf_values.append(fv_dp)
    
    volume_fraction_pdf_values_array = np.array(volume_fraction_pdf_values)
    sum_of_pdf_values = np.sum(volume_fraction_pdf_values_array)
    
    normalized_volume_fraction = np.zeros_like(volume_fraction_pdf_values_array)
    if sum_of_pdf_values > 1e-9:
        normalized_volume_fraction = volume_fraction_pdf_values_array / sum_of_pdf_values

    cumulative_volume_undersize = np.cumsum(normalized_volume_fraction)
    cumulative_volume_oversize = 1 - cumulative_volume_undersize

    plot_data['dp_values_ft'] = dp_values_ft
    plot_data['volume_fraction'] = normalized_volume_fraction
    plot_data['cumulative_volume_undersize'] = cumulative_volume_undersize
    plot_data['cumulative_volume_oversize'] = cumulative_volume_oversize

    # --- New E and Entrained Flow Calculations ---
    Ug_si = inputs['V_g_input']
    Q_liquid_mass_flow_rate_input_si = inputs['Q_liquid_mass_flow_rate_input']
    rho_l_input_si = inputs['rho_l_input']
    
    Wl_for_e_calc = Q_liquid_mass_flow_rate_input_si
    
    E_fraction = calculate_e_interpolated(Ug_si, Wl_for_e_calc)
    
    Q_entrained_total_mass_flow_rate_si = E_fraction * Q_liquid_mass_flow_rate_input_si
    
    Q_entrained_total_volume_flow_rate_si = 0.0
    if rho_l_input_si > 0:
        Q_entrained_total_volume_flow_rate_si = Q_entrained_total_mass_flow_rate_si / rho_l_input_si

    results['Wl_for_e_calc'] = Wl_for_e_calc
    results['E_fraction'] = E_fraction
    results['Q_entrained_total_mass_flow_rate_si'] = Q_entrained_total_mass_flow_rate_si
    results['Q_entrained_total_volume_flow_rate_si'] = Q_entrained_total_volume_flow_rate_si

    entrained_mass_flow_rate_per_dp = [
        fv_norm * Q_entrained_total_mass_flow_rate_si for fv_norm in normalized_volume_fraction
    ]
    plot_data['entrained_mass_flow_rate_per_dp'] = entrained_mass_flow_rate_per_dp

    entrained_volume_flow_rate_per_dp = [
        fv_norm * Q_entrained_total_volume_flow_rate_si for fv_norm in normalized_volume_fraction
    ]
    plot_data['entrained_volume_flow_rate_per_dp'] = entrained_volume_flow_rate_per_dp

    return results, plot_data


# Initialize session state for inputs and results if not already present
if 'inputs' not in st.session_state:
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
        'Q_liquid_mass_flow_rate_input': 0.1, # New input: kg/s (example value)
        'num_points_distribution': 20, # Default number of points
    }
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
    mass_flow_unit = "kg/s"

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
    
    st.session_state.inputs['Q_liquid_mass_flow_rate_input'] = st.number_input(f"Total Liquid Mass Flow Rate ({mass_flow_unit})", min_value=0.0, value=st.session_state.inputs['Q_liquid_mass_flow_rate_input'], format="%.2f", key='Q_liquid_mass_flow_rate_input_widget',
                                help="The total mass flow rate of the liquid phase entering the system. This value is directly used as 'Wl' in the entrainment calculation.")

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

    # When inputs on this page change, trigger recalculation for initial state
    import datetime
    st.session_state.report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        st.session_state.calculation_results, st.session_state.plot_data = _calculate_all_data(st.session_state.inputs)
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
        mass_flow_unit = "kg/s"
        vol_flow_unit = "m³/s" # New unit for Streamlit display

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
        st.write(f"Total Liquid Mass Flow Rate: {st.session_state.inputs['Q_liquid_mass_flow_rate_input']:.2f} {mass_flow_unit}") # New input
        st.write(f"Number of Points for Distribution: {st.session_state.inputs['num_points_distribution']}") # Display here
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

        # Step 6: Entrainment Fraction (E) Calculation
        st.markdown("#### Step 6: Calculate Entrainment Fraction (E)")
        st.write(f"Gas Velocity (Ug): {st.session_state.inputs['V_g_input']:.2f} {vel_unit}")
        st.write(f"Liquid Loading (Wl): {st.session_state.inputs['Q_liquid_mass_flow_rate_input']:.2f} {mass_flow_unit}")
        st.success(f"**Result:** Entrainment Fraction (E) = **{results['E_fraction']:.4f}** (dimensionless)")
        st.success(f"**Result:** Total Entrained Liquid Mass Flow Rate = **{results['Q_entrained_total_mass_flow_rate_si']:.4f} {mass_flow_unit}**")
        st.success(f"**Result:** Total Entrained Liquid Volume Flow Rate = **{results['Q_entrained_total_volume_flow_rate_si']:.6f} {vol_flow_unit}**") # New total volume flow
        st.markdown("---")

        st.info("Step 7 (Generating Droplet Size Distribution Data and Entrained Flow per size) is performed internally to prepare data for the plot and table.")

    else:
        st.warning("Please go to the 'Input Parameters' page and modify inputs to trigger calculations.")

# --- Page: Droplet Distribution Results ---
elif page == "Droplet Distribution Results":
    st.header("3. Particle Size Distribution Plot")

    # Moved the input for num_points_distribution here
    st.subheader("Distribution Plot Settings")
    st.session_state.inputs['num_points_distribution'] = st.number_input(
        "Number of Points for Distribution Plot/Table",
        min_value=10,
        max_value=100,
        value=st.session_state.inputs['num_points_distribution'],
        step=5,
        key='num_points_distribution_input',
        help="Adjust the number of data points used to generate the droplet size distribution curve and table (10-100)."
    )
    st.markdown("---") # Add a separator after the input

    # Recalculate plot_data specifically on this page after num_points_distribution is updated
    try:
        st.session_state.calculation_results, st.session_state.plot_data = _calculate_all_data(
            st.session_state.inputs,
            num_points_override=st.session_state.inputs['num_points_distribution']
        )
    except Exception as e:
        st.error(f"An error occurred during plot data calculation: {e}")
        st.session_state.calculation_results = None
        st.session_state.plot_data = None


    if st.session_state.plot_data:
        plot_data = st.session_state.plot_data
        
        # Define unit labels for plotting
        micron_unit_label = "µm" # Always SI for this version
        mass_flow_unit = "kg/s"
        vol_flow_unit = "m³/s" # New unit for Streamlit display

        dp_values_microns = plot_data['dp_values_ft'] * FT_TO_MICRON
        # Use normalized_volume_fraction for plotting the distribution curve as per article's definition
        volume_fraction_for_plot = plot_data['volume_fraction'] # This is now the normalized one
        cumulative_volume_undersize = plot_data['cumulative_volume_undersize']
        cumulative_volume_oversize = plot_data['cumulative_volume_oversize'] 
        entrained_mass_flow_rate_per_dp = plot_data['entrained_mass_flow_rate_per_dp'] # New data
        entrained_volume_flow_rate_per_dp = plot_data['entrained_volume_flow_rate_per_dp'] # New data

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(dp_values_microns, cumulative_volume_undersize, 'o-', label='Cumulative Volume Undersize', markersize=2, color='#1f77b4')
        ax1.plot(dp_values_microns, cumulative_volume_oversize, 'o-', label='Cumulative Volume Oversize', markersize=2, color='#d62728')
        ax1.set_xlabel(f'Droplet Size ({micron_unit_label})', fontsize=12)
        ax1.set_ylabel('Cumulative Volume Fraction', color='black', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_ylim(0, 1.05) # Adjusted y-limit for cumulative fractions
        ax1.set_xlim(0, max(dp_values_microns) * 1.1 if dp_values_microns.size > 0 else 1000)

        ax2 = ax1.twinx()
        # Plot normalized volume fraction on the right axis
        ax2.plot(dp_values_microns, volume_fraction_for_plot, 'o-', label='Volume/Mass Fraction', markersize=2, color='#2ca02c')
        ax2.set_ylabel('Volume/Mass Fraction', color='black', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Adjust y-limit for normalized volume/mass fraction (it now sums to 1)
        max_norm_fv = max(volume_fraction_for_plot) if volume_fraction_for_plot.size > 0 else 0.1
        ax2.set_ylim(0, max_norm_fv * 1.2)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=10)

        plt.title('Entrainment Droplet Size Distribution', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # --- Volume Fraction Data Table for Streamlit App ---
        st.subheader("Volume Fraction Data Table (All Points)")
        if dp_values_microns.size > 0:
            # Display all data points based on user's input
            full_df = pd.DataFrame({
                "Droplet Size (µm)": dp_values_microns,
                "Volume Fraction": volume_fraction_for_plot,
                "Cumulative Undersize": cumulative_volume_undersize,
                f"Entrained Mass Flow ({mass_flow_unit})": entrained_mass_flow_rate_per_dp,
                f"Entrained Volume Flow ({vol_flow_unit})": entrained_volume_flow_rate_per_dp # New column
            })
            st.dataframe(full_df.style.format({
                "Droplet Size (µm)": "{:.2f}",
                "Volume Fraction": "{:.4f}",
                "Cumulative Undersize": "{:.4f}",
                f"Entrained Mass Flow ({mass_flow_unit})": "{:.6f}",
                f"Entrained Volume Flow ({vol_flow_unit})": "{:.9f}" # Format for new column
            }))
            
            # Display sum check for verification
            st.markdown(f"**Sum of Entrained Mass Flow in Table:** {np.sum(entrained_mass_flow_rate_per_dp):.6f} {mass_flow_unit}")
            st.markdown(f"**Total Entrained Liquid Mass Flow Rate (Step 6):** {st.session_state.calculation_results['Q_entrained_total_mass_flow_rate_si']:.6f} {mass_flow_unit}")
            st.markdown(f"**Sum of Entrained Volume Flow in Table:** {np.sum(entrained_volume_flow_rate_per_dp):.9f} {vol_flow_unit}") # New sum check
            st.markdown(f"**Total Entrained Liquid Volume Flow Rate (Step 6):** {st.session_state.calculation_results['Q_entrained_total_volume_flow_rate_si']:.9f} {vol_flow_unit}") # New sum check
            st.info("Note: The sum of 'Entrained Flow' in the table should now precisely match the 'Total Entrained Liquid Flow Rate' from Step 6, as the volume frequency distribution is normalized and all calculated points are displayed.")

        else:
            st.info("No data available to display in the table. Please check your input parameters.")

        # Save plot to a BytesIO object for PDF embedding
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        buf.seek(0) # Rewind to the beginning of the buffer

        # Prepare data for PDF table (all points based on user's input)
        plot_data_for_pdf_table = {
            'dp_values_microns': dp_values_microns,
            'volume_fraction': volume_fraction_for_plot, # Pass normalized for PDF table
            'cumulative_volume_undersize': cumulative_volume_undersize,
            'cumulative_volume_oversize': cumulative_volume_oversize, # Keep this in plot_data for internal consistency if needed elsewhere, but not used in PDF table
            'entrained_mass_flow_rate_per_dp': entrained_mass_flow_rate_per_dp,
            'entrained_volume_flow_rate_per_dp': entrained_volume_flow_rate_per_dp # Pass new data
        }

        st.download_button(
            label="Download Report as PDF",
            data=generate_pdf_report(st.session_state.inputs, st.session_state.calculation_results, buf, plot_data_for_pdf_table),
            file_name="Droplet_Distribution_Report.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Please go to the 'Input Parameters' page and modify inputs to trigger calculations and generate the plot data.")

st.markdown(r"""
---
#### Important Notes:
* **Figure 9 Approximation:** The "Droplet Size Distribution Shift Factor" is based on a simplified interpretation of Figure 9 from the article. For highly precise engineering applications, the curves in Figure 9 would need to be digitized and accurately modeled.
* **Log Normal Distribution Parameters:** The article states typical values for $a=4.0$ and $\delta=0.72$. The formula for $\delta$ shown in the article ($\delta=\frac{0.394}{log(\frac{V_{so}}{V_{so}})}$) appears to be a typographical error, so the constant value $\delta=0.72$ is used as indicated in the text.
* **Units:** This application now exclusively uses the International System (SI) units for all inputs and outputs. All internal calculations are still performed in FPS units to align with the article's correlations, with automatic conversions handled internally.
* **Entrainment Fraction (E) Calculation:** As per your instruction, the 'Wl' parameter in the empirical equations for Entrainment Fraction (E) is now directly the 'Total Liquid Mass Flow Rate' (kg/s) provided by the user. Linear interpolation is applied for Ug values between the defined points. The accuracy of these correlations is dependent on the range and conditions for which they were originally developed.
* **Volume/Mass Fraction Distribution:** The 'Volume Fraction' and 'Mass Fraction' values displayed in the table and plot now represent the **normalized volume frequency distribution** as stated in the article. This means the sum of these fractions over the entire distribution range is 1. Consequently, the sum of 'Entrained Flow (kg/s)' for sampled points in the table will approximate the 'Total Entrained Liquid Mass Flow Rate' calculated in Step 6, with the full sum matching if all data points were included.
""")
