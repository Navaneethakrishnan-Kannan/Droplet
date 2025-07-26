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
IN_TO_FT = 1/12 # 1 inch = 1/12 feet

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
    elif unit_type == "pressure": # psig to psi (psig is already a unit of pressure)
        return value # No conversion needed for psig to psi, just use the value directly
    elif unit_type == "diameter_in": # inches to feet
        return value * IN_TO_FT
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

# Figure 6: C_d vs Re_p data (digitized from plot)
CD_VS_REP_DATA = {
    "Re_p": np.array([
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0,
        2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 30.0, 40.0, 50.0,
        70.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 700.0, 1000.0, 2000.0,
        5000.0, 10000.0, 100000.0, 200000.0, 500000.0, 1000000.0
    ]),
    "Cd": np.array([
        25000, 12500, 5000, 2500, 1250, 500, 250, 125, 50, 25,
        12.5, 8.3, 6.25, 5.0, 3.5, 2.5, 1.5, 1.0, 0.8, 0.7,
        0.6, 0.5, 0.45, 0.42, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
        0.4, 0.4, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65
    ])
}

def calculate_terminal_velocity(dp_fps, rho_l_fps, rho_g_fps, mu_g_fps, g_fps=32.174):
    """
    Calculates the terminal settling velocity of a droplet using an iterative approach
    based on Eq. 2, Eq. 3, and Figure 6 (Cd vs Re_p).
    All inputs and outputs are in FPS units.
    g_fps: acceleration due to gravity in ft/s^2
    Returns: Vt_new, Cd, Re_p
    """
    if rho_g_fps == 0 or mu_g_fps == 0 or (rho_l_fps - rho_g_fps) <= 0:
        return 0.0, 0.0, 0.0 # Prevent division by zero or non-physical density difference

    # Initial guess for Vt (e.g., using Stokes' Law for small droplets)
    # This initial guess helps the iteration converge faster for typical values.
    # If dp_fps is very small, Stokes' Law is a good start.
    if dp_fps > 0:
        Vt_guess = (g_fps * dp_fps**2 * (rho_l_fps - rho_g_fps)) / (18 * mu_g_fps)
    else:
        Vt_guess = 0.0

    Vt_current = Vt_guess
    tolerance = 1e-6
    max_iterations = 100
    
    Re_p = 0.0
    Cd = 0.0

    for _ in range(max_iterations):
        if Vt_current <= 0 or dp_fps <= 0: # Handle cases where velocity or diameter is zero/negative
            Re_p = 0.0
        else:
            Re_p = (dp_fps * Vt_current * rho_g_fps) / mu_g_fps

        # Get Cd from Re_p using interpolation from Figure 6 data
        Cd = np.interp(Re_p, CD_VS_REP_DATA["Re_p"], CD_VS_REP_DATA["Cd"])
        
        # Ensure Cd is not zero or negative
        if Cd <= 0:
            Cd = 0.01 # Small positive value to avoid division by zero, or handle as error

        # Calculate new Vt using Eq. 2
        # Ensure the argument inside sqrt is non-negative
        arg_sqrt = (4 * g_fps * dp_fps * (rho_l_fps - rho_g_fps)) / (3 * Cd * rho_g_fps)
        if arg_sqrt < 0:
            Vt_new = 0.0 # Cannot have imaginary velocity
        else:
            Vt_new = arg_sqrt**0.5
        
        if abs(Vt_new - Vt_current) < tolerance:
            return Vt_new, Cd, Re_p
        
        Vt_current = Vt_new
    
    # If max_iterations reached without convergence, return the last calculated value and warn
    st.warning(f"Terminal velocity calculation did not converge for dp={dp_fps*FT_TO_MICRON:.2f} um after {max_iterations} iterations. Returning last value: {Vt_current:.6f} ft/s.")
    return Vt_current, Cd, Re_p


# Figure 2: F, Actual Velocity/Average (Plug Flow) Velocity vs. L/Di (digitized from plot)
F_FACTOR_DATA = {
    "No inlet device": {
        "L_over_Di": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "F_value": np.array([3.0, 2.5, 2.1, 1.8, 1.5, 1.3, 1.2, 1.1, 1.05, 1.02, 1.0])
    },
    "Diverter plate": {
        "L_over_Di": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "F_value": np.array([2.0, 1.7, 1.5, 1.4, 1.3, 1.2, 1.15, 1.1, 1.05, 1.02, 1.0])
    },
    "Half-pipe": {
        "L_over_Di": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "F_value": np.array([1.4, 1.3, 1.25, 1.2, 1.15, 1.1, 1.08, 1.05, 1.03, 1.02, 1.0])
    },
    "Vane-type": {
        "L_over_Di": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "F_value": np.array([1.3, 1.2, 1.15, 1.1, 1.08, 1.05, 1.03, 1.02, 1.01, 1.0, 1.0])
    },
    "Cyclonic": {
        "L_over_Di": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "F_value": np.array([1.2, 1.1, 1.05, 1.03, 1.02, 1.01, 1.0, 1.0, 1.0, 1.0])
    }
}

def get_f_factor(inlet_device, L_over_Di, has_perforated_plate):
    """
    Calculates the F factor (Actual Velocity/Average (Plug Flow) Velocity)
    using linear interpolation based on the inlet device and L/Di from Figure 2.
    Applies perforated plate adjustment if selected.
    """
    if inlet_device not in F_FACTOR_DATA:
        st.warning(f"Unknown inlet device for F-factor: '{inlet_device}'. Defaulting F factor to 1.0.")
        return 1.0

    data = F_FACTOR_DATA[inlet_device]
    x_values = data["L_over_Di"]
    y_values = data["F_value"]

    # Clamp L_over_Di to the defined range
    clamped_L_over_Di = max(min(L_over_Di, x_values.max()), x_values.min())

    f_value = np.interp(clamped_L_over_Di, x_values, y_values)

    if has_perforated_plate:
        # Apply perforated plate adjustment: F_effective = F - 0.5 * (F - 1)
        f_value_adjusted = f_value - 0.5 * (f_value - 1)
        # Ensure F_value_adjusted is not less than 1.0 (perfect plug flow)
        return float(max(1.0, f_value_adjusted))
    
    return float(f_value)


# Table 3: Mesh Pad K Deration Factors as a Function of Pressure
K_DERATION_DATA = {
    "pressure_psig": np.array([0, 100, 200, 400, 600, 800, 1000, 1200]),
    "k_factor_percent": np.array([100, 93, 88, 83, 80, 78, 76, 75])
}

def get_k_deration_factor(pressure_psig):
    """
    Calculates the K deration factor based on pressure using linear interpolation from Table 3.
    """
    if pressure_psig < K_DERATION_DATA["pressure_psig"].min():
        # Clamp to min pressure, use max K factor
        return K_DERATION_DATA["k_factor_percent"].max() / 100.0
    elif pressure_psig > K_DERATION_DATA["pressure_psig"].max():
        # Clamp to max pressure, use min K factor
        return K_DERATION_DATA["k_factor_percent"].min() / 100.0

    k_factor_percent = np.interp(pressure_psig, K_DERATION_DATA["pressure_psig"], K_DERATION_DATA["k_factor_percent"])
    return float(k_factor_percent / 100.0)

# Table 2: Mesh Pad Design and Construction Parameters (FPS units for internal use)
MESH_PAD_PARAMETERS = {
    "Standard mesh pad": {
        "density_lb_ft3": 9,
        "voidage_percent": 98.5,
        "wire_diameter_in": 0.011,
        "specific_surface_area_ft2_ft3": 85,
        "Ks_ft_sec": 0.35,
        "liquid_load_gal_min_ft2": 0.75,
        "thickness_in": 6 # Typical thickness as per Fig 9 example
    },
    "High-capacity mesh pad": {
        "density_lb_ft3": 5,
        "voidage_percent": 99.0,
        "wire_diameter_in": 0.011,
        "specific_surface_area_ft2_ft3": 45,
        "Ks_ft_sec": 0.4,
        "liquid_load_gal_min_ft2": 1.5,
        "thickness_in": 6
    },
    "High-efficiency co-knit mesh pad": {
        "density_lb_ft3": 12,
        "voidage_percent": 96.2,
        "wire_diameter_in": 0.011, # Assuming 0.011 x 0.0008 means effective wire diameter is 0.011
        "specific_surface_area_ft2_ft3": 83, # Using 83, not 1100, as 1100 seems like a typo for a different unit
        "Ks_ft_sec": 0.25,
        "liquid_load_gal_min_ft2": 0.5,
        "thickness_in": 6
    }
}

# Table 4: Vane-Pack Design and Construction Parameters (FPS units for internal use)
VANE_PACK_PARAMETERS = {
    "Simple vane": { # Assuming upflow as default, horizontal is a variant
        "flow_direction": "Upflow", # This is a choice, not a fixed parameter
        "number_of_bends": 5, # Using 5-8, pick 5 as a representative
        "vane_spacing_in": 0.75, # Using 0.5-1, pick 0.75 as a representative
        "bend_angle_degree": 45, # Using 30-60, pick 45 as common
        "Ks_ft_sec_upflow": 0.5, # From table
        "Ks_ft_sec_horizontal": 0.65, # From table
        "liquid_load_gal_min_ft2": 2
    },
    "High-capacity pocketed vane": {
        "flow_direction": "Upflow", # This is a choice, not a fixed parameter
        "number_of_bends": 5,
        "vane_spacing_in": 0.75,
        "bend_angle_degree": 45,
        "Ks_ft_sec_upflow": 0.82, # Using 0.82-1.15, pick 0.82
        "Ks_ft_sec_horizontal": 0.82, # Using 0.82-1.15, pick 0.82
        "liquid_load_gal_min_ft2": 5
    }
}

# Table 5: Typical Demisting Axial-Flow Cyclone Design and Construction Parameters (FPS units for internal use)
CYCLONE_PARAMETERS = {
    "2.0 in. cyclones": { # Only one type given, so this is the default
        "cyclone_inside_diameter_in": 2.0,
        "cyclone_length_in": 10,
        "inlet_swirl_angle_degree": 45,
        "cyclone_to_cyclone_spacing_diameters": 1.75,
        "Ks_ft_sec_bundle_face_area": 0.8, # Using ~0.8-1, pick 0.8
        "liquid_load_gal_min_ft2_bundle_face_area": 10
    }
}


# Figure 8: Single-wire droplet capture efficiency (Ew) vs. Stokes' number (Stk)
# Curve fit given by Eq. 13: Ew = (-0.105 + 0.995 * Stk^1.00493) / (0.6261 + Stk^1.00493)
def calculate_single_wire_efficiency(Stk):
    """Calculates single-wire impaction efficiency using Eq. 13."""
    if Stk <= 0: # Handle Stk=0 or negative to avoid math domain errors
        return 0.0
    # CORRECTED: Changed Stk exponent in numerator from 0.0493 to 1.00493 based on provided image
    numerator = -0.105 + 0.995 * (Stk**1.00493)
    denominator = 0.6261 + (Stk**1.00493)
    if denominator == 0: # Avoid division by zero
        return 0.0
    Ew = numerator / denominator
    return max(0.0, min(1.0, Ew)) # Ensure efficiency is between 0 and 1

# --- Mist Extractor Efficiency Functions ---

def mesh_pad_efficiency_func(dp_fps, V_g_eff_sep_fps, rho_l_fps, rho_g_fps, mu_g_fps, mesh_pad_type_params_fps):
    """
    Calculates the droplet removal efficiency for a mesh pad using Equations 12, 13, and 14.
    All inputs in FPS units.
    Returns: E_pad, Stk, Ew (for detailed reporting)
    """
    if V_g_eff_sep_fps <= 0 or mu_g_fps <= 0 or dp_fps <= 0:
        return 0.0, 0.0, 0.0 # No impaction if no gas flow or zero droplet/gas viscosity

    Dw_fps = mesh_pad_type_params_fps["wire_diameter_in"] * IN_TO_FT
    pad_thickness_fps = mesh_pad_type_params_fps["thickness_in"] * IN_TO_FT
    specific_surface_area_fps = mesh_pad_type_params_fps["specific_surface_area_ft2_ft3"]
    
    # Eq. 12: Stokes' number
    # Note: Article states some literature uses 9 in denominator instead of 18. Using 18 as per Eq. 12.
    if Dw_fps == 0: return 0.0, 0.0, 0.0 # Avoid division by zero if wire diameter is zero
    Stk = ((rho_l_fps - rho_g_fps) * (dp_fps**2) * V_g_eff_sep_fps) / (18 * mu_g_fps * Dw_fps)

    # Eq. 13: Single-wire capture efficiency
    Ew = calculate_single_wire_efficiency(Stk)

    # Eq. 14: Mesh-pad removal efficiency
    # Note: The article's Eq. 14 is E_pad = 1 - e^(-0.0238 * S * T * Ew) (typo in article, should be -0.0238*S*T*Ew)
    # Based on Carpenter and Othmer (1955), the exponent should be - (some constant) * S * T * Ew
    # The constant 0.0238 is for FPS units.
    exponent = -0.0238 * specific_surface_area_fps * pad_thickness_fps * Ew
    E_pad = 1 - np.exp(exponent)
    
    return max(0.0, min(1.0, E_pad)), Stk, Ew # Ensure efficiency is between 0 and 1

def vane_type_efficiency_func(dp_fps, V_g_eff_sep_fps, rho_l_fps, rho_g_fps, mu_g_fps, vane_type_params_fps):
    """
    Calculates the droplet separation efficiency for a vane-type mist extractor using Eq. 15.
    All inputs in FPS units.
    Returns: E_vane, Stk (dummy), Ew (dummy) - for consistency in _calculate_and_apply_separation
    """
    if V_g_eff_sep_fps <= 0 or mu_g_fps <= 0 or dp_fps <= 0:
        return 0.0, 0.0, 0.0

    num_bends = vane_type_params_fps["number_of_bends"]
    vane_spacing_fps = vane_type_params_fps["vane_spacing_in"] * IN_TO_FT
    bend_angle_rad = np.deg2rad(vane_type_params_fps["bend_angle_degree"])

    # Eq. 15: Evane = 1 - exp[ - (n * dp^3 * (rho_l - rho_g) * Vg_eff_sep) / (515.7 * mu_g * b * cos^2(theta)) ]
    # Note: The article's Eq. 15 is slightly ambiguous with the exponent. Assuming it's a single term.
    # Also, the dp is cubed in the numerator, which is unusual for a Stokes-like number.
    # Assuming Vg_eff_sep is the gas velocity through the vane pack.
    
    numerator = num_bends * (dp_fps**3) * (rho_l_fps - rho_g_fps) * V_g_eff_sep_fps
    denominator = 515.7 * mu_g_fps * vane_spacing_fps * (np.cos(bend_angle_rad)**2)

    if denominator == 0:
        return 0.0, 0.0, 0.0

    exponent = - (numerator / denominator)
    E_vane = 1 - np.exp(exponent)

    # Return dummy Stk and Ew for non-mesh pad types for consistent function signature
    return max(0.0, min(1.0, E_vane)), 0.0, 0.0

def demisting_cyclone_efficiency_func(dp_fps, V_g_eff_sep_fps, rho_l_fps, rho_g_fps, mu_g_fps, cyclone_type_params_fps):
    """
    Calculates the droplet removal efficiency for an individual axial-flow cyclone tube
    using Eq. 16 and the associated Stokes' number definition.
    All inputs in FPS units.
    Returns: E_cycl, Stk_cycl, Ew (dummy) - for consistency in _calculate_and_apply_separation
    """
    if V_g_eff_sep_fps <= 0 or mu_g_fps <= 0 or dp_fps <= 0:
        return 0.0, 0.0, 0.0

    Dcycl_fps = cyclone_type_params_fps["cyclone_inside_diameter_in"] * IN_TO_FT
    Lcycl_fps = cyclone_type_params_fps["cyclone_length_in"] * IN_TO_FT
    in_swirl_angle_rad = np.deg2rad(cyclone_type_params_fps["inlet_swirl_angle_degree"])

    # Eq. 16: Stk_cycl = ( (rho_l - rho_g) * dp^2 * Vg_cycl ) / (18 * mu_g * Dcycl)
    # Vg_cycl is superficial gas velocity through a single cyclone tube.
    # Assuming V_g_eff_sep_fps is the superficial velocity through the cyclone bundle face area,
    # and this can be used as Vg_cycl for a single cyclone for efficiency calculation.
    Vg_cycl = V_g_eff_sep_fps # Approximation for simplicity as bundle area is not easily translated to single tube area without more info.

    if Dcycl_fps == 0: return 0.0, 0.0, 0.0 # Avoid division by zero
    Stk_cycl = ((rho_l_fps - rho_g_fps) * (dp_fps**2) * Vg_cycl) / (18 * mu_g_fps * Dcycl_fps)

    # Eq. 16: E_cycl = 1 - exp[ -8 * Stk_cycl * (Lcycl / (Dcycl * tan(alpha))) ]
    # Ensure tan(alpha) is not zero or near zero for 90 degree swirl angle etc.
    if np.tan(in_swirl_angle_rad) == 0:
        return 0.0, 0.0, 0.0 # No swirl, no separation
    
    exponent = -8 * Stk_cycl * (Lcycl_fps / (Dcycl_fps * np.tan(in_swirl_angle_rad)))
    E_cycl = 1 - np.exp(exponent)

    # Return dummy Ew for non-mesh pad types for consistent function signature
    return max(0.0, min(1.0, E_cycl)), Stk_cycl, 0.0


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
        
    def add_table(self, headers, data, col_widths, title=None):
        if title:
            self.set_font('Arial', 'B', 10)
            self.cell(0, 7, title, 0, 1, 'L')
            self.ln(2)

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


def generate_pdf_report(inputs, results, plot_image_buffer_original, plot_image_buffer_adjusted, plot_data_original, plot_data_adjusted, plot_data_after_gravity, plot_data_after_mist_extractor):
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
    
    sigma_display_val = inputs['sigma_custom'] # Directly use sigma_custom for display
    pdf.chapter_body(f"Liquid Surface Tension (sigma): {sigma_display_val:.3f} N/m")
    pdf.chapter_body(f"Selected Inlet Device: {inputs['inlet_device']}")
    pdf.chapter_body(f"Total Liquid Mass Flow Rate: {inputs['Q_liquid_mass_flow_rate_input']:.2f} kg/s") # New input
    pdf.chapter_body(f"Number of Points for Distribution: {inputs['num_points_distribution']}") # New input
    pdf.ln(5)

    pdf.chapter_body(f"Separator Type: {inputs['separator_type']}")
    if inputs['separator_type'] == "Horizontal":
        pdf.chapter_body(f"Gas Space Height (hg): {inputs['h_g_input']:.3f} m")
        pdf.chapter_body(f"Effective Separation Length (Le): {inputs['L_e_input']:.3f} m")
    else: # Vertical
        pdf.chapter_body(f"Separator Diameter: {inputs['D_separator_input']:.3f} m")
    
    pdf.chapter_body(f"Length from Inlet Device to Mist Extractor (L_to_ME): {inputs['L_to_ME_input']:.3f} m")
    pdf.chapter_body(f"Perforated Plate Used: {'Yes' if inputs['perforated_plate_option'] else 'No'}")
    pdf.chapter_body(f"Operating Pressure: {inputs['pressure_psig_input']:.1f} psig")
    pdf.ln(5)

    pdf.chapter_body(f"Mist Extractor Type: {inputs['mist_extractor_type']}")
    if inputs['mist_extractor_type'] == "Mesh Pad":
        pdf.chapter_body(f"  Mesh Pad Type: {inputs['mesh_pad_type']}")
        pdf.chapter_body(f"  Mesh Pad Thickness: {inputs['mesh_pad_thickness_in']:.2f} in")
    elif inputs['mist_extractor_type'] == "Vane-Type":
        pdf.chapter_body(f"  Vane Type: {inputs['vane_type']}")
        pdf.chapter_body(f"  Flow Direction: {inputs['vane_flow_direction']}")
        pdf.chapter_body(f"  Number of Bends: {inputs['vane_num_bends']}")
        pdf.chapter_body(f"  Vane Spacing: {inputs['vane_spacing_in']:.2f} in")
        pdf.chapter_body(f"  Bend Angle: {inputs['vane_bend_angle_deg']:.1f} deg")
    elif inputs['mist_extractor_type'] == "Cyclonic":
        pdf.chapter_body(f"  Cyclone Type: {inputs['cyclone_type']}")
        pdf.chapter_body(f"  Cyclone Diameter: {inputs['cyclone_diameter_in']:.2f} in")
        pdf.chapter_body(f"  Cyclone Length: {inputs['cyclone_length_in']:.2f} in")
        pdf.chapter_body(f"  Inlet Swirl Angle: {inputs['cyclone_swirl_angle_deg']:.1f} deg")
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
    pressure_unit_pdf = "psig"

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
    pdf.chapter_body(f"  Operating Pressure: {inputs['pressure_psig_input']:.1f} {pressure_unit_pdf}")
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
    pdf.chapter_body(f"Calculation (FPS): rho_g V_g^2 = {to_fps(inputs['rho_g_input'], 'density'):.4f} lb/ft^3 * ({to_fps(inputs['V_g_input'], 'velocity'):.2f}^2) = {results['rho_v_squared_fps']:.2f} lb/ft-sec^2")
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
    
    pdf.chapter_body(f"For **Original** $d_{{v50}}$:")
    pdf.chapter_body(f"Equation: $d_{{max, original}} = a \\cdot d_{{v50, original}}$")
    pdf.chapter_body(f"Calculation (FPS): $d_{{max, original}} = {A_DISTRIBUTION} \\cdot {results['dv50_original_fps']:.6f} \\text{{ ft}} = {results['d_max_original_fps']:.6f} \\text{{ ft}}$")
    pdf.chapter_body(f"Result: Maximum Droplet Size (Original $d_{{max}}$) = {results['d_max_original_fps'] * FT_TO_MICRON:.2f} {micron_unit_label_pdf} ({from_fps(results['d_max_original_fps'], 'length'):.6f} {len_unit_pdf})")
    pdf.ln(2) # Small line break for readability

    pdf.chapter_body(f"For **Adjusted** $d_{{v50}}$:")
    pdf.chapter_body(f"Equation: $d_{{max, adjusted}} = a \\cdot d_{{v50, adjusted}}$")
    pdf.chapter_body(f"Calculation (FPS): $d_{{max, adjusted}} = {A_DISTRIBUTION} \\cdot {results['dv50_adjusted_fps']:.6f} \\text{{ ft}} = {results['d_max_adjusted_fps']:.6f} \\text{{ ft}}$")
    pdf.chapter_body(f"Result: Maximum Droplet Size (Adjusted $d_{{max}}$) = {results['d_max_adjusted_fps'] * FT_TO_MICRON:.2f} {micron_unit_label_pdf} ({from_fps(results['d_max_adjusted_fps'], 'length'):.6f} {len_unit_pdf})")
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

    # Step 7: Calculate F-factor and Effective Gas Velocity
    pdf.set_font('Arial', 'B', 10)
    pdf.chapter_body("Step 7: Calculate F-factor and Effective Gas Velocity in Separator")
    pdf.set_font('Arial', '', 10)
    pdf.chapter_body(f"L/Di Ratio (L_to_ME / D_pipe): {results['L_over_Di']:.2f}")
    pdf.chapter_body(f"Inlet Device: {inputs['inlet_device']}")
    pdf.chapter_body(f"Perforated Plate Used: {'Yes' if inputs['perforated_plate_option'] else 'No'}")
    pdf.chapter_body(f"Calculated F-factor: {results['F_factor']:.3f}")
    pdf.chapter_body(f"Effective Gas Velocity in Separator (V_g_effective_separator): {from_fps(results['V_g_effective_separator_fps'], 'velocity'):.2f} {vel_unit_pdf}")
    pdf.ln(5)

    # Debug print statement for PDF generation context
    print(f"PDF Gen: Length of gravity_details_table_data: {len(plot_data_after_gravity['gravity_details_table_data']) if plot_data_after_gravity and 'gravity_details_table_data' in plot_data_after_gravity else 'N/A'}")

    # Display detailed table for gravity separation
    if plot_data_after_gravity and plot_data_after_gravity['gravity_details_table_data']:
        pdf.set_font('Arial', 'B', 10)
        # Add a new page if the table might overflow
        if pdf.get_y() + 10 + (len(plot_data_after_gravity['gravity_details_table_data']) + 1) * 6 > pdf.page_break_trigger:
            pdf.add_page()
            pdf.chapter_title('2. Step-by-Step Calculation Results (Continued)') # Add a continued title
            pdf.ln(5) # Some space after continued title

        pdf.add_table(
            headers=["Droplet Size (um)", "Vt (ft/s)", "Cd", "Re_p", "Flow Regime", "Time Settle (s)", "h_max_settle (ft)", "Edp"],
            data=[
                [
                    f"{row_dict['dp_microns']:.2f}",
                    f"{row_dict['Vt_ftps']:.4f}",
                    f"{row_dict['Cd']:.4f}",
                    f"{row_dict['Re_p']:.2e}",
                    row_dict['Flow Regime'],
                    f"{row_dict['Time Settle (s)']:.4f}",
                    f"{row_dict['h_max_settle (ft)']:.4f}",
                    f"{row_dict['Edp']:.2%}"
                ] for row_dict in plot_data_after_gravity['gravity_details_table_data']
            ],
            col_widths=[25, 20, 15, 20, 25, 25, 25, 15], # Adjust these widths as needed
            title='Detailed Droplet Separation Performance in Gas Gravity Section'
        )
    else:
        pdf.chapter_body("Detailed droplet separation data for gravity section not available.")
    pdf.ln(5) # Add spacing after the table or message

    # Step 9: Mist Extractor Performance
    pdf.set_font('Arial', 'B', 10)
    pdf.chapter_body("Step 9: Mist Extractor Performance")
    pdf.set_font('Arial', '', 10)
    pdf.chapter_body(f"Mist Extractor Type: {inputs['mist_extractor_type']}")
    pdf.chapter_body(f"Operating Pressure: {inputs['pressure_psig_input']:.1f} {pressure_unit_pdf}")
    pdf.chapter_body(f"K-Deration Factor (from Table 3): {results['k_deration_factor']:.3f}")

    if inputs['mist_extractor_type'] == "Mesh Pad":
        pdf.chapter_body(f"  Mesh Pad Type: {inputs['mesh_pad_type']}")
        pdf.chapter_body(f"  Mesh Pad Thickness: {inputs['mesh_pad_thickness_in']:.2f} in")
        pdf.chapter_body(f"  Wire Diameter: {results['mesh_pad_params']['wire_diameter_in']:.3f} in")
        pdf.chapter_body(f"  Specific Surface Area: {results['mesh_pad_params']['specific_surface_area_ft2_ft3']:.1f} ft^2/ft^3")
        pdf.chapter_body(f"  Base K_s: {results['mesh_pad_params']['Ks_ft_sec']:.2f} ft/sec")
        pdf.chapter_body(f"  Liquid Load Capacity: {results['mesh_pad_params']['liquid_load_gal_min_ft2']:.2f} gal/min/ft^2")
    elif inputs['mist_extractor_type'] == "Vane-Type":
        pdf.chapter_body(f"  Vane Type: {inputs['vane_type']}")
        pdf.chapter_body(f"  Flow Direction: {inputs['vane_flow_direction']}")
        pdf.chapter_body(f"  Number of Bends: {inputs['vane_num_bends']}")
        pdf.chapter_body(f"  Vane Spacing: {inputs['vane_spacing_in']:.2f} in")
        pdf.chapter_body(f"  Bend Angle: {inputs['vane_bend_angle_deg']:.1f} deg")
        pdf.chapter_body(f"  Base K_s (Upflow): {results['vane_type_params']['Ks_ft_sec_upflow']:.2f} ft/sec")
        pdf.chapter_body(f"  Base K_s (Horizontal): {results['vane_type_params']['Ks_ft_sec_horizontal']:.2f} ft/sec")
        pdf.chapter_body(f"  Liquid Load Capacity: {results['vane_type_params']['liquid_load_gal_min_ft2']:.2f} gal/min/ft^2")
    elif inputs['mist_extractor_type'] == "Cyclonic":
        pdf.chapter_body(f"  Cyclone Type: {inputs['cyclone_type']}")
        pdf.chapter_body(f"  Cyclone Diameter: {inputs['cyclone_diameter_in']:.2f} in")
        pdf.chapter_body(f"  Cyclone Length: {inputs['cyclone_length_in']:.2f} in")
        pdf.chapter_body(f"  Inlet Swirl Angle: {inputs['cyclone_swirl_angle_deg']:.1f} deg")
        pdf.chapter_body(f"  Base K_s: {results['cyclone_type_params']['Ks_ft_sec_bundle_face_area']:.2f} ft/sec")
        pdf.chapter_body(f"  Liquid Load Capacity: {results['cyclone_type_params']['liquid_load_gal_min_ft2_bundle_face_area']:.2f} gal/min/ft^2")
    
    pdf.chapter_body(f"Overall Separation Efficiency of Mist Extractor: {results['mist_extractor_separation_efficiency']:.2%}")
    pdf.chapter_body(f"Total Entrained Liquid Mass Flow Rate After Mist Extractor: {plot_data_after_mist_extractor['total_entrained_mass_flow_rate_si']:.4f} {mass_flow_unit_pdf}")
    pdf.chapter_body(f"Total Entrained Liquid Volume Flow Rate After Mist Extractor: {plot_data_after_mist_extractor['total_entrained_volume_flow_rate_si']:.6f} {vol_flow_unit_pdf}")
    pdf.ln(5)

    # Display detailed table for mist extractor performance
    if plot_data_after_mist_extractor and plot_data_after_mist_extractor['mist_extractor_details_table_data']:
        pdf.set_font('Arial', 'B', 10)
        if pdf.get_y() + 10 + (len(plot_data_after_mist_extractor['mist_extractor_details_table_data']) + 1) * 6 > pdf.page_break_trigger:
            pdf.add_page()
            pdf.chapter_title('2. Step-by-Step Calculation Results (Continued)')
            pdf.ln(5)

        if inputs['mist_extractor_type'] == "Mesh Pad":
            pdf.add_table(
                headers=["Droplet Size (um)", "Stokes' No.", "Single-Wire Eff. (Ew)", "Mesh-Pad Eff. (E_pad)"],
                data=[
                    [
                        f"{row_dict['dp_microns']:.2f}",
                        f"{row_dict['Stk']:.2e}",
                        f"{row_dict['Ew']:.4f}",
                        f"{row_dict['E_pad']:.2%}"
                    ] for row_dict in plot_data_after_mist_extractor['mist_extractor_details_table_data']
                ],
                col_widths=[30, 30, 40, 40],
                title=f'Detailed Droplet Separation Performance in {inputs["mist_extractor_type"]} Mist Extractor'
            )
        elif inputs['mist_extractor_type'] == "Vane-Type":
             pdf.add_table(
                headers=["Droplet Size (um)", "Vane Eff. (E_vane)"],
                data=[
                    [
                        f"{row_dict['dp_microns']:.2f}",
                        f"{row_dict['E_vane']:.2%}"
                    ] for row_dict in plot_data_after_mist_extractor['mist_extractor_details_table_data']
                ],
                col_widths=[30, 40],
                title=f'Detailed Droplet Separation Performance in {inputs["mist_extractor_type"]} Mist Extractor'
            )
        elif inputs['mist_extractor_type'] == "Cyclonic":
            pdf.add_table(
                headers=["Droplet Size (um)", "Stokes' No. (Cycl)", "Cyclone Eff. (E_cycl)"],
                data=[
                    [
                        f"{row_dict['dp_microns']:.2f}",
                        f"{row_dict['Stk_cycl']:.2e}",
                        f"{row_dict['E_cycl']:.2%}"
                    ] for row_dict in plot_data_after_mist_extractor['mist_extractor_details_table_data']
                ],
                col_widths=[30, 40, 40],
                title=f'Detailed Droplet Separation Performance in {inputs["mist_extractor_type"]} Mist Extractor'
            )
    else:
        pdf.chapter_body("Detailed droplet separation data for mist extractor not available.")
    pdf.ln(5)


    pdf.set_font('Arial', 'B', 10)
    pdf.chapter_body("Final Carry-Over from Separator Outlet:")
    pdf.set_font('Arial', '', 10)
    pdf.chapter_body(f"  Total Carry-Over Mass Flow Rate: {plot_data_after_mist_extractor['total_entrained_mass_flow_rate_si']:.4f} {mass_flow_unit_pdf}")
    pdf.chapter_body(f"  Total Carry-Over Volume Flow Rate: {plot_data_after_mist_extractor['total_entrained_volume_flow_rate_si']:.6f} {vol_flow_unit_pdf}")
    pdf.ln(5)


    # --- Droplet Distribution Plots ---
    pdf.add_page() # Start a new page for the plots
    pdf.chapter_title('3. Droplet Distribution Results')
    
    pdf.chapter_body("The following graphs show the calculated entrainment droplet size distribution:")

    # Original Distribution Plot
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, '3.1. Distribution Before Inlet Device', 0, 1, 'L')
    pdf.ln(2)
    if plot_image_buffer_original:
        pdf.image(plot_image_buffer_original, x=10, y=pdf.get_y(), w=pdf.w - 20)
    pdf.ln(5)

    # Adjusted Distribution Plot (after inlet device)
    pdf.add_page() # Ensure the second plot is on a new page
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, '3.2. Distribution After Inlet Device (Shift Factor Applied)', 0, 1, 'L')
    pdf.ln(2)
    if plot_image_buffer_adjusted:
        pdf.image(plot_image_buffer_adjusted, x=10, y=pdf.get_y(), w=pdf.w - 20)
    pdf.ln(5)

    # Distribution After Gravity Settling Plot
    pdf.add_page() # Ensure this plot is on a new page
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, '3.3. Distribution After Gas Gravity Settling', 0, 1, 'L')
    pdf.ln(2)
    # Generate and add plot for after gravity settling
    if plot_data_after_gravity and plot_data_after_gravity['dp_values_ft'].size > 0:
        fig_after_gravity, ax_after_gravity = plt.subplots(figsize=(10, 6))
        dp_values_microns_after_gravity = plot_data_after_gravity['dp_values_ft'] * FT_TO_MICRON
        
        ax_after_gravity.plot(dp_values_microns_after_gravity, plot_data_after_gravity['cumulative_volume_undersize'], 'o-', label='Cumulative Volume Undersize', markersize=2, color='#1f77b4')
        ax_after_gravity.plot(dp_values_microns_after_gravity, plot_data_after_gravity['cumulative_volume_oversize'], 'o-', label='Cumulative Volume Oversize', markersize=2, color='#d62728')
        ax_after_gravity.set_xlabel(f'Droplet Size ({micron_unit_label_pdf})', fontsize=12)
        ax_after_gravity.set_ylabel('Cumulative Volume Fraction', color='black', fontsize=12)
        ax_after_gravity.tick_params(axis='y', labelcolor='black')
        ax_after_gravity.set_ylim(0, 1.05)
        ax_after_gravity.set_xlim(0, max(dp_values_microns_after_gravity) * 1.1 if dp_values_microns_after_gravity.size > 0 else 1000)

        ax2_after_gravity = ax_after_gravity.twinx()
        ax2_after_gravity.plot(dp_values_microns_after_gravity, plot_data_after_gravity['volume_fraction'], 'o-', label='Volume/Mass Fraction', markersize=2, color='#2ca02c')
        ax2_after_gravity.set_ylabel('Volume/Mass Fraction', color='black', fontsize=12)
        ax2_after_gravity.tick_params(axis='y', labelcolor='black')
        max_norm_fv_after_gravity = max(plot_data_after_gravity['volume_fraction']) if plot_data_after_gravity['volume_fraction'].size > 0 else 0.1
        ax2_after_gravity.set_ylim(0, max_norm_fv_after_gravity * 1.2)

        lines_after_gravity, labels_after_gravity = ax_after_gravity.get_legend_handles_labels()
        lines2_after_gravity, labels2_after_gravity = ax2_after_gravity.get_legend_handles_labels()
        ax2_after_gravity.legend(lines_after_gravity + lines2_after_gravity, labels_after_gravity + labels2_after_gravity, loc='upper left', fontsize=10)

        plt.title('Entrainment Droplet Size Distribution (After Gravity Settling)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        buf_after_gravity = io.BytesIO()
        fig_after_gravity.savefig(buf_after_gravity, format="png", dpi=300)
        buf_after_gravity.seek(0)
        pdf.image(buf_after_gravity, x=10, y=pdf.get_y(), w=pdf.w - 20)
        pdf.ln(5)
        plt.close(fig_after_gravity) # Close the plot to free memory
    else:
        pdf.chapter_body("No data available for distribution after gravity settling. Please check your input parameters.")

    # Distribution After Mist Extractor Plot
    pdf.add_page() # Ensure this plot is on a new page
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, '3.4. Distribution After Mist Extractor', 0, 1, 'L')
    pdf.ln(2)
    if plot_data_after_mist_extractor and plot_data_after_mist_extractor['dp_values_ft'].size > 0:
        fig_after_me, ax_after_me = plt.subplots(figsize=(10, 6))
        dp_values_microns_after_me = plot_data_after_mist_extractor['dp_values_ft'] * FT_TO_MICRON
        
        ax_after_me.plot(dp_values_microns_after_me, plot_data_after_mist_extractor['cumulative_volume_undersize'], 'o-', label='Cumulative Volume Undersize', markersize=2, color='#1f77b4')
        ax_after_me.plot(dp_values_microns_after_me, plot_data_after_mist_extractor['cumulative_volume_oversize'], 'o-', label='Cumulative Volume Oversize', markersize=2, color='#d62728')
        ax_after_me.set_xlabel(f'Droplet Size ({micron_unit_label_pdf})', fontsize=12)
        ax_after_me.set_ylabel('Cumulative Volume Fraction', color='black', fontsize=12)
        ax_after_me.tick_params(axis='y', labelcolor='black')
        ax_after_me.set_ylim(0, 1.05)
        ax_after_me.set_xlim(0, max(dp_values_microns_after_me) * 1.1 if dp_values_microns_after_me.size > 0 else 1000)

        ax2_after_me = ax_after_me.twinx()
        ax2_after_me.plot(dp_values_microns_after_me, plot_data_after_mist_extractor['volume_fraction'], 'o-', label='Volume/Mass Fraction', markersize=2, color='#2ca02c')
        ax2_after_me.set_ylabel('Volume/Mass Fraction', color='black', fontsize=12)
        ax2_after_me.tick_params(axis='y', labelcolor='black')
        max_norm_fv_after_me = max(plot_data_after_mist_extractor['volume_fraction']) if plot_data_after_mist_extractor['volume_fraction'].size > 0 else 0.1
        ax2_after_me.set_ylim(0, max_norm_fv_after_me * 1.2)

        lines_after_me, labels_after_me = ax_after_me.get_legend_handles_labels()
        lines2_after_me, labels2_after_me = ax2_after_me.get_legend_handles_labels()
        ax2_after_me.legend(lines_after_me + lines2_after_me, labels_after_me + labels2_after_me, loc='upper left', fontsize=10)

        plt.title('Entrainment Droplet Size Distribution (After Mist Extractor)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        buf_after_me = io.BytesIO()
        fig_after_me.savefig(buf_after_me, format="png", dpi=300)
        buf_after_me.seek(0)
        pdf.image(buf_after_me, x=10, y=pdf.get_y(), w=pdf.w - 20)
        pdf.ln(5)
        plt.close(fig_after_me) # Close the plot to free memory
    else:
        pdf.chapter_body("No data available for distribution after mist extractor. Please check your input parameters.")


    # --- Volume Fraction Data Tables ---
    pdf.add_page() # Start a new page for the tables
    pdf.chapter_title('4. Volume Fraction Data Tables (Sampled)')

    # Original Data Table
    if plot_data_original and 'dp_values_ft' in plot_data_original and len(plot_data_original['dp_values_ft']) > 0:
        headers = ["Droplet Size (um)", "Volume Fraction", "Cumulative Undersize", "Entrained Mass Flow (kg/s)", "Entrained Volume Flow (m^3/s)"]
        
        # Original Data Table
        full_data_original = []
        for i in range(len(plot_data_original['dp_values_ft'])): # Iterate using 'dp_values_ft'
            full_data_original.append([
                f"{plot_data_original['dp_values_ft'][i] * FT_TO_MICRON:.2f}", # Convert to microns here
                f"{plot_data_original['volume_fraction'][i]:.4f}",
                f"{plot_data_original['cumulative_volume_undersize'][i]:.4f}",
                f"{plot_data_original['entrained_mass_flow_rate_per_dp'][i]:.6f}",
                f"{plot_data_original['entrained_volume_flow_rate_per_dp'][i]:.9f}"
            ])
        col_widths = [25, 25, 35, 40, 45]
        pdf.add_table(headers, full_data_original, col_widths, title='4.1. Distribution Before Inlet Device')
    else:
        pdf.chapter_body("No data available for original distribution table. Please check your input parameters.")
    
    # Adjusted Data Table
    if plot_data_adjusted and 'dp_values_ft' in plot_data_adjusted and len(plot_data_adjusted['dp_values_ft']) > 0:
        # Adjusted Data Table
        full_data_adjusted = []
        for i in range(len(plot_data_adjusted['dp_values_ft'])): # Iterate using 'dp_values_ft'
            full_data_adjusted.append([
                f"{plot_data_adjusted['dp_values_ft'][i] * FT_TO_MICRON:.2f}", # Convert to microns here
                f"{plot_data_adjusted['volume_fraction'][i]:.4f}",
                f"{plot_data_adjusted['cumulative_volume_undersize'][i]:.4f}",
                f"{plot_data_adjusted['entrained_mass_flow_rate_per_dp'][i]:.6f}",
                f"{plot_data_adjusted['entrained_volume_flow_rate_per_dp'][i]:.9f}"
            ])
        col_widths = [25, 25, 35, 40, 45]
        pdf.add_table(headers, full_data_adjusted, col_widths, title='4.2. Distribution After Inlet Device (Shift Factor Applied)')
    else:
        pdf.chapter_body("No data available for adjusted distribution table. Please check your input parameters.")

    # Data Table After Gravity Settling
    if plot_data_after_gravity and 'dp_values_ft' in plot_data_after_gravity and len(plot_data_after_gravity['dp_values_ft']) > 0:
        full_data_after_gravity = []
        for i in range(len(plot_data_after_gravity['dp_values_ft'])):
            full_data_after_gravity.append([
                f"{plot_data_after_gravity['dp_values_ft'][i] * FT_TO_MICRON:.2f}",
                f"{plot_data_after_gravity['volume_fraction'][i]:.4f}",
                f"{plot_data_after_gravity['cumulative_volume_undersize'][i]:.4f}",
                f"{plot_data_after_gravity['entrained_mass_flow_rate_per_dp'][i]:.6f}",
                f"{plot_data_after_gravity['entrained_volume_flow_rate_per_dp'][i]:.9f}"
            ])
        col_widths = [25, 25, 35, 40, 45]
        pdf.add_table(headers, full_data_after_gravity, col_widths, title='4.3. Distribution After Gas Gravity Settling')
    else:
        pdf.chapter_body("No data available for distribution table after gravity settling. Please check your input parameters.")

    # Data Table After Mist Extractor
    if plot_data_after_mist_extractor and 'dp_values_ft' in plot_data_after_mist_extractor and len(plot_data_after_mist_extractor['dp_values_ft']) > 0:
        full_data_after_me = []
        for i in range(len(plot_data_after_mist_extractor['dp_values_ft'])):
            full_data_after_me.append([
                f"{plot_data_after_mist_extractor['dp_values_ft'][i] * FT_TO_MICRON:.2f}",
                f"{plot_data_after_mist_extractor['volume_fraction'][i]:.4f}",
                f"{plot_data_after_mist_extractor['cumulative_volume_undersize'][i]:.4f}",
                f"{plot_data_after_mist_extractor['entrained_mass_flow_rate_per_dp'][i]:.6f}",
                f"{plot_data_after_mist_extractor['entrained_volume_flow_rate_per_dp'][i]:.9f}"
            ])
        col_widths = [25, 25, 35, 40, 45]
        pdf.add_table(headers, full_data_after_me, col_widths, title='4.4. Distribution After Mist Extractor')
    else:
        pdf.chapter_body("No data available for distribution table after mist extractor. Please check your input parameters.")


    return bytes(pdf.output(dest='S')) # Return PDF as bytes directly

# --- Streamlit App Layout ---

st.set_page_config(layout="centered", page_title="Oil & Gas Separation App")

st.title("🛢️ Oil and Gas Separation: Particle Size Distribution")
st.markdown("""
This application helps quantify the entrained liquid droplet size distribution in gas-liquid separators,
based on the principles and correlations discussed in the article "Quantifying Separation Performance" by Mark Bothamley.
All inputs and outputs are in **SI Units**.
""")

# --- Helper function to generate distribution data for a given dv50 and d_max ---
def _generate_initial_distribution_data(dv50_value_fps, d_max_value_fps, num_points, E_fraction, Q_liquid_mass_flow_rate_si, rho_l_input_si):
    """
    Generates initial particle size distribution data (volume fraction, cumulative, entrained flow)
    for a given dv50, d_max, and other flow parameters.
    """
    plot_data = {
        'dp_values_ft': np.array([]),
        'volume_fraction': np.array([]),
        'cumulative_volume_undersize': np.array([]),
        'cumulative_volume_oversize': np.array([]),
        'entrained_mass_flow_rate_per_dp': np.array([]),
        'entrained_volume_flow_rate_per_dp': np.array([]),
        'total_entrained_mass_flow_rate_si': 0.0,
        'total_entrained_volume_flow_rate_si': 0.0
    }

    # Add a check for valid dv50 and d_max values
    if dv50_value_fps <= 0 or d_max_value_fps <= 0 or d_max_value_fps < dv50_value_fps:
        st.warning(f"Warning: Invalid droplet size range calculated (dv50: {dv50_value_fps:.6f} ft, d_max: {d_max_value_fps:.6f} ft). Check input parameters.")
        return plot_data # Return empty data

    dp_min_calc_fps = dv50_value_fps * 0.01
    dp_max_calc_fps = d_max_value_fps * 0.999

    if dp_min_calc_fps >= dp_max_calc_fps:
        st.warning("Calculated min droplet size is greater than or equal to max droplet size. Distribution cannot be generated.")
        return plot_data # Return empty data

    dp_values_ft = np.linspace(dp_min_calc_fps, dp_max_calc_fps, num_points)
    
    volume_fraction_pdf_values = []

    for dp in dp_values_ft:
        if dp >= d_max_value_fps:
            z_val = np.inf
        else:
            # Ensure the argument to log is positive and finite
            log_arg = (A_DISTRIBUTION * dp) / (d_max_value_fps - dp)
            if log_arg <= 0: # Handle cases where dp is too small or d_max_value_fps - dp is too small
                z_val = -np.inf
            else:
                z_val = np.log(log_arg)
        
        # Handle potential division by zero for fv_dp if dp or (d_max_value_fps - dp) is zero
        if dp == 0 or (d_max_value_fps - dp) == 0:
            fv_dp = 0
        else:
            denominator = np.sqrt(np.pi * dp * (d_max_value_fps - dp))
            if denominator == 0:
                fv_dp = 0
            else:
                fv_dp = ((DELTA_DISTRIBUTION * d_max_value_fps) / denominator) * np.exp(-DELTA_DISTRIBUTION**2 * z_val**2)
        
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

    # Calculate total entrained liquid mass and volume flow rates
    Q_entrained_total_mass_flow_rate_si = E_fraction * Q_liquid_mass_flow_rate_si
    Q_entrained_total_volume_flow_rate_si = 0.0
    if rho_l_input_si > 0:
        Q_entrained_total_volume_flow_rate_si = Q_entrained_total_mass_flow_rate_si / rho_l_input_si

    # Calculate entrained mass flow rate per droplet size interval using the normalized volume fraction
    plot_data['entrained_mass_flow_rate_per_dp'] = [
        fv_norm * Q_entrained_total_mass_flow_rate_si for fv_norm in normalized_volume_fraction
    ]

    # Calculate entrained volume flow rate per droplet size interval
    plot_data['entrained_volume_flow_rate_per_dp'] = [
        fv_norm * Q_entrained_total_volume_flow_rate_si for fv_norm in normalized_volume_fraction
    ]
    plot_data['total_entrained_mass_flow_rate_si'] = Q_entrained_total_mass_flow_rate_si
    plot_data['total_entrained_volume_flow_rate_si'] = Q_entrained_total_volume_flow_rate_si

    return plot_data


def _calculate_and_apply_separation(
    initial_plot_data,
    separation_stage_efficiency_func=None, # Function to apply for separation
    is_gravity_stage=False, # Flag to indicate if this is the gravity separation stage
    V_g_eff_sep_fps=0.0, # Required for gravity and mist extractor
    rho_l_fps=0.0, rho_g_fps=0.0, mu_g_fps=0.0, # Required for terminal velocity calc
    h_g_sep_fps=0.0, L_e_sep_fps=0.0, # Required for horizontal gravity
    separator_type="Horizontal", # Required for gravity
    mist_extractor_type_str="", # Added to differentiate mist extractor types
    **kwargs_for_efficiency_func # Arguments for the efficiency function
):
    """
    Applies a separation efficiency function to an existing droplet distribution
    and calculates the new entrained mass/volume flow rates.
    If is_gravity_stage is True, it also collects detailed per-droplet data.
    """
    if not initial_plot_data or not initial_plot_data['dp_values_ft'].size > 0:
        return {
            'dp_values_ft': np.array([]),
            'volume_fraction': np.array([]),
            'cumulative_volume_undersize': np.array([]),
            'cumulative_volume_oversize': np.array([]),
            'entrained_mass_flow_rate_per_dp': np.array([]),
            'entrained_volume_flow_rate_per_dp': np.array([]),
            'total_entrained_mass_flow_rate_si': 0.0,
            'total_entrained_volume_flow_rate_si': 0.0,
            'overall_separation_efficiency': 0.0,
            'gravity_details_table_data': [], # Added for detailed gravity data
            'mist_extractor_details_table_data': [] # Added for detailed mist extractor data
        }

    dp_values_ft = initial_plot_data['dp_values_ft']
    initial_volume_fraction = initial_plot_data['volume_fraction']
    initial_entrained_mass_flow_rate_per_dp = initial_plot_data['entrained_mass_flow_rate_per_dp']
    initial_entrained_volume_flow_rate_per_dp = initial_plot_data['entrained_volume_flow_rate_per_dp']

    separated_entrained_mass_flow_rate_per_dp = np.zeros_like(initial_entrained_mass_flow_rate_per_dp)
    separated_entrained_volume_flow_rate_per_dp = np.zeros_like(initial_entrained_volume_flow_rate_per_dp)
    
    # Calculate initial total entrained flow rates from the provided initial_plot_data
    initial_total_entrained_mass_flow_rate_si = np.sum(initial_entrained_mass_flow_rate_per_dp)
    initial_total_entrained_volume_flow_rate_si = np.sum(initial_entrained_volume_flow_rate_per_dp)

    gravity_details_table_data = [] # To store details for Step 8 table
    mist_extractor_details_table_data = [] # To store details for Step 9 table

    # Apply separation efficiency for each droplet size
    for i, dp in enumerate(dp_values_ft):
        efficiency = 0.0
        
        # Variables for gravity details table
        Vt = 0.0
        Cd = 0.0
        Re_p = 0.0
        flow_regime = "N/A"
        time_settle = 0.0 # Time for droplet to fall h_g or L_e
        h_max_settle = 0.0 # Max height droplet can fall in gas residence time (horizontal) or effective separation height (vertical)

        # Variables for mist extractor details table
        Stk_me = 0.0 # Stokes number for mist extractor
        Ew_me = 0.0 # Single-wire efficiency for mesh pad
        E_pad_me = 0.0 # Mesh pad efficiency
        Stk_cycl_me = 0.0 # Stokes number for cyclone

        if separation_stage_efficiency_func:
            # For gravity stage, we need more detailed returns from the efficiency function
            if is_gravity_stage:
                if separator_type == "Horizontal":
                    efficiency, Vt, Cd, Re_p, h_max_settle_calc = gravity_efficiency_func_horizontal(
                        dp_fps=dp, V_g_eff_sep_fps=V_g_eff_sep_fps, h_g_sep_fps=h_g_sep_fps,
                        L_e_sep_fps=L_e_sep_fps, rho_l_fps=rho_l_fps, rho_g_fps=rho_g_fps, mu_g_fps=mu_g_fps
                    )
                    # Time for droplet to fall h_g
                    if Vt > 1e-9: # Avoid division by zero
                        time_settle = h_g_sep_fps / Vt
                    else:
                        time_settle = float('inf')
                    h_max_settle = h_max_settle_calc # This is the h_max_settle calculated within the function
                else: # Vertical
                    efficiency, Vt, Cd, Re_p = gravity_efficiency_func_vertical(
                        dp_fps=dp, V_g_eff_sep_fps=V_g_eff_sep_fps, rho_l_fps=rho_l_fps,
                        rho_g_fps=rho_g_fps, mu_g_fps=mu_g_fps
                    )
                    # For vertical, h_max_settle is effectively the height of the gas gravity section if separated
                    # Time for droplet to fall L_e (gas gravity section height)
                    if Vt > 1e-9:
                        time_settle = L_e_sep_fps / Vt
                    else:
                        time_settle = float('inf')
                    h_max_settle = L_e_sep_fps if efficiency > 0 else 0.0 # If separated, it effectively settles through L_e
                
                # Determine flow regime
                if Re_p < 2:
                    flow_regime = "Stokes'"
                elif 2 <= Re_p <= 500:
                    flow_regime = "Intermediate"
                else:
                    flow_regime = "Newton's"

                gravity_details_table_data.append({
                    "dp_microns": dp * FT_TO_MICRON,
                    "Vt_ftps": Vt,
                    "Cd": Cd,
                    "Re_p": Re_p,
                    "Flow Regime": flow_regime,
                    "Time Settle (s)": time_settle,
                    "h_max_settle (ft)": h_max_settle,
                    "Edp": efficiency # Individual droplet efficiency
                })

            else: # For mist extractor stage (collect details based on type)
                if mist_extractor_type_str == "Mesh Pad":
                    E_pad_me, Stk_me, Ew_me = separation_stage_efficiency_func(
                        dp_fps=dp,
                        V_g_eff_sep_fps=V_g_eff_sep_fps,
                        rho_l_fps=rho_l_fps,
                        rho_g_fps=rho_g_fps,
                        mu_g_fps=mu_g_fps,
                        **kwargs_for_efficiency_func
                    )
                    efficiency = E_pad_me
                    mist_extractor_details_table_data.append({
                        "dp_microns": dp * FT_TO_MICRON,
                        "Stk": Stk_me,
                        "Ew": Ew_me,
                        "E_pad": E_pad_me
                    })
                elif mist_extractor_type_str == "Vane-Type":
                    E_vane_me, _, _ = separation_stage_efficiency_func( # Vane function returns dummy Stk, Ew
                        dp_fps=dp,
                        V_g_eff_sep_fps=V_g_eff_sep_fps,
                        rho_l_fps=rho_l_fps,
                        rho_g_fps=rho_g_fps,
                        mu_g_fps=mu_g_fps,
                        **kwargs_for_efficiency_func
                    )
                    efficiency = E_vane_me
                    mist_extractor_details_table_data.append({
                        "dp_microns": dp * FT_TO_MICRON,
                        "E_vane": E_vane_me
                    })
                elif mist_extractor_type_str == "Cyclonic":
                    E_cycl_me, Stk_cycl_me, _ = separation_stage_efficiency_func( # Cyclone function returns dummy Ew
                        dp_fps=dp,
                        V_g_eff_sep_fps=V_g_eff_sep_fps,
                        rho_l_fps=rho_l_fps,
                        rho_g_fps=rho_g_fps,
                        mu_g_fps=mu_g_fps,
                        **kwargs_for_efficiency_func
                    )
                    efficiency = E_cycl_me
                    mist_extractor_details_table_data.append({
                        "dp_microns": dp * FT_TO_MICRON,
                        "Stk_cycl": Stk_cycl_me,
                        "E_cycl": E_cycl_me
                    })
                else: # Fallback if mist_extractor_type_str is not recognized
                    efficiency = 0.0
            
            # Ensure efficiency is between 0 and 1
            efficiency = max(0.0, min(1.0, efficiency))
            
            separated_entrained_mass_flow_rate_per_dp[i] = initial_entrained_mass_flow_rate_per_dp[i] * (1 - efficiency)
            separated_entrained_volume_flow_rate_per_dp[i] = initial_entrained_volume_flow_rate_per_dp[i] * (1 - efficiency)
        else:
            # If no separation function, just carry over the initial values
            separated_entrained_mass_flow_rate_per_dp[i] = initial_entrained_mass_flow_rate_per_dp[i]
            separated_entrained_volume_flow_rate_per_dp[i] = initial_entrained_volume_flow_rate_per_dp[i]

    # Calculate new total entrained flow rates after this separation stage
    final_total_entrained_mass_flow_rate_si = np.sum(separated_entrained_mass_flow_rate_per_dp)
    final_total_entrained_volume_flow_rate_si = np.sum(separated_entrained_volume_flow_rate_per_dp)

    # Calculate overall separation efficiency for this stage
    overall_separation_efficiency = 0.0
    if initial_total_entrained_mass_flow_rate_si > 1e-9: # Avoid division by near-zero
        overall_separation_efficiency = 1.0 - (final_total_entrained_mass_flow_rate_si / initial_total_entrained_mass_flow_rate_si)

    # Recalculate normalized volume fraction based on the remaining entrained mass flow
    # This represents the *new* distribution of the *remaining* droplets
    new_volume_fraction = np.zeros_like(separated_entrained_mass_flow_rate_per_dp)
    if final_total_entrained_mass_flow_rate_si > 1e-9:
        new_volume_fraction = separated_entrained_mass_flow_rate_per_dp / final_total_entrained_mass_flow_rate_si
    
    new_cumulative_volume_undersize = np.cumsum(new_volume_fraction)
    new_cumulative_volume_oversize = 1 - new_cumulative_volume_undersize

    return {
        'dp_values_ft': dp_values_ft,
        'volume_fraction': new_volume_fraction, # This is the new normalized distribution of remaining droplets
        'cumulative_volume_undersize': new_cumulative_volume_undersize,
        'cumulative_volume_oversize': new_cumulative_volume_oversize,
        'entrained_mass_flow_rate_per_dp': separated_entrained_mass_flow_rate_per_dp,
        'entrained_volume_flow_rate_per_dp': separated_entrained_volume_flow_rate_per_dp,
        'total_entrained_mass_flow_rate_si': final_total_entrained_mass_flow_rate_si,
        'total_entrained_volume_flow_rate_si': final_total_entrained_volume_flow_rate_si,
        'overall_separation_efficiency': overall_separation_efficiency,
        'gravity_details_table_data': gravity_details_table_data, # Include detailed data for gravity stage
        'mist_extractor_details_table_data': mist_extractor_details_table_data # Include detailed data for mist extractor stage
    }


# --- Gravity Settling Efficiency Functions ---
def gravity_efficiency_func_horizontal(dp_fps, V_g_eff_sep_fps, h_g_sep_fps, L_e_sep_fps, rho_l_fps, rho_g_fps, mu_g_fps):
    """
    Calculates separation efficiency for a horizontal separator's gas gravity section.
    Assumes uniform droplet release over h_g.
    Returns: efficiency, Vt, Cd, Re_p, h_max_settle
    """
    if V_g_eff_sep_fps <= 0 or h_g_sep_fps <= 0 or L_e_sep_fps <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0 # No separation if no gas flow or no settling height/length

    V_t, Cd, Re_p = calculate_terminal_velocity(dp_fps, rho_l_fps, rho_g_fps, mu_g_fps)
    
    # Calculate maximum height from which a droplet can settle
    if V_g_eff_sep_fps > 1e-9: # Avoid division by near zero
        h_max_settle = (V_t * L_e_sep_fps) / V_g_eff_sep_fps
    else:
        h_max_settle = float('inf') # If gas velocity is zero, droplet can settle from infinite height

    # Efficiency is the fraction of h_g from which droplets of this size will settle
    efficiency = min(1.0, h_max_settle / h_g_sep_fps)
    return efficiency, V_t, Cd, Re_p, h_max_settle

def gravity_efficiency_func_vertical(dp_fps, V_g_eff_sep_fps, rho_l_fps, rho_g_fps, mu_g_fps):
    """
    Calculates separation efficiency for a vertical separator's gas gravity section.
    Sharp cutoff: 100% if Vt > V_g_eff_sep, 0% otherwise.
    Returns: efficiency, Vt, Cd, Re_p
    """
    V_t, Cd, Re_p = calculate_terminal_velocity(dp_fps, rho_l_fps, rho_g_fps, mu_g_fps)
    
    if V_t > V_g_eff_sep_fps:
        efficiency = 1.0 # Droplet settles
    else:
        efficiency = 0.0 # Droplet is carried over
    
    # For vertical, h_max_settle is conceptually the entire gravity section height if it settles
    # We'll handle this detail in the _calculate_and_apply_separation function for table consistency
    return efficiency, V_t, Cd, Re_p


# --- Function to perform all main calculations ---
def _perform_main_calculations(inputs):
    """Performs all scalar calculations and returns results dictionary."""
    results = {}

    # Convert all SI inputs to FPS for consistent calculation
    D_pipe_fps = to_fps(inputs['D_input'], "length")
    rho_l_fps = to_fps(inputs['rho_l_input'], "density")
    mu_l_fps = to_fps(inputs['mu_l_input'], "viscosity")
    V_g_input_fps = to_fps(inputs['V_g_input'], "velocity") # Superficial gas velocity in feed pipe
    rho_g_fps = to_fps(inputs['rho_g_input'], "density")
    mu_g_fps = to_fps(inputs['mu_g_input'], "viscosity")
    sigma_fps = inputs['sigma_fps'] # This is already in poundal/ft
    
    # Separator specific inputs
    L_to_ME_fps = to_fps(inputs['L_to_ME_input'], 'length')
    D_separator_fps = to_fps(inputs['D_separator_input'], 'length')
    h_g_input_fps = to_fps(inputs['h_g_input'], 'length')
    L_e_input_fps = to_fps(inputs['L_e_input'], 'length')
    
    # Step 1: Calculate Superficial Gas Reynolds Number (Re_g) in feed pipe
    if mu_g_fps == 0: raise ValueError("Gas viscosity (μg) cannot be zero for Reynolds number calculation.")
    Re_g = (D_pipe_fps * V_g_input_fps * rho_g_fps) / mu_g_fps
    results['Re_g'] = Re_g

    # Step 2: Calculate Volume Median Diameter ($d_{v50}$) without inlet device effect
    if V_g_input_fps == 0 or rho_g_fps == 0 or rho_l_fps == 0 or mu_l_fps == 0:
        raise ValueError("Gas velocity, gas density, liquid density, and liquid viscosity must be non-zero for $d_{v50}$ calculation.")
    
    dv50_original_fps = 0.01 * (sigma_fps / (rho_g_fps * V_g_input_fps**2)) * (Re_g**(2/3)) * ((rho_g_fps / rho_l_fps)**(-1/3)) * ((mu_g_fps / mu_l_fps)**(2/3))
    results['dv50_original_fps'] = dv50_original_fps
    
    # Calculate d_max for the original distribution
    d_max_original_fps = A_DISTRIBUTION * dv50_original_fps
    results['d_max_original_fps'] = d_max_original_fps


    # Step 3: Determine Inlet Momentum (rho_g V_g^2)
    rho_v_squared_fps = rho_g_fps * V_g_input_fps**2
    results['rho_v_squared_fps'] = rho_v_squared_fps

    # Step 4: Apply Inlet Device "Droplet Size Distribution Shift Factor"
    shift_factor = get_shift_factor(inputs['inlet_device'], rho_v_squared_fps)
    dv50_adjusted_fps = dv50_original_fps * shift_factor
    results['shift_factor'] = shift_factor
    results['dv50_adjusted_fps'] = dv50_adjusted_fps

    # Step 5: Calculate parameters for Upper-Limit Log Normal Distribution for adjusted dv50
    d_max_adjusted_fps = A_DISTRIBUTION * dv50_adjusted_fps
    results['d_max_adjusted_fps'] = d_max_adjusted_fps

    # Step 6: Entrainment Fraction (E) Calculation
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

    # Step 7: Calculate F-factor and Effective Gas Velocity in Separator
    L_over_Di = L_to_ME_fps / D_pipe_fps
    F_factor = get_f_factor(inputs['inlet_device'], L_over_Di, inputs['perforated_plate_option'])
    results['L_over_Di'] = L_over_Di
    results['F_factor'] = F_factor

    V_g_effective_separator_fps = 0.0
    if inputs['separator_type'] == "Vertical":
        # For vertical, gas velocity in separator is (Q_g_feed / A_separator_gas)
        # Q_g_feed = V_g_input_fps (feed pipe velocity) * A_pipe_fps
        A_pipe_fps = np.pi * (D_pipe_fps / 2)**2
        A_separator_gas_vertical_fps = np.pi * (D_separator_fps / 2)**2
        if A_separator_gas_vertical_fps > 0:
            V_g_superficial_separator_fps = (V_g_input_fps * A_pipe_fps) / A_separator_gas_vertical_fps
            V_g_effective_separator_fps = V_g_superficial_separator_fps / F_factor
        else:
            raise ValueError("Separator diameter cannot be zero for vertical separator gas velocity calculation.")
    else: # Horizontal
        # For horizontal, assume V_g_input is superficial velocity in separator gas section
        # and F_factor adjusts it.
        V_g_superficial_separator_fps = V_g_input_fps # Assuming V_g_input is now superficial in separator for horizontal
        V_g_effective_separator_fps = V_g_superficial_separator_fps / F_factor
    
    results['V_g_effective_separator_fps'] = V_g_effective_separator_fps

    # Step 8: Gas Gravity Separation Section Efficiency
    # This will be used to calculate plot_data_after_gravity
    gravity_separation_efficiency = 0.0
    if inputs['separator_type'] == "Horizontal":
        if V_g_effective_separator_fps <= 0 or h_g_input_fps <= 0 or L_e_input_fps <= 0:
            st.warning("Invalid horizontal separator dimensions for gravity settling calculation. Efficiency set to 0.")
            gravity_separation_efficiency = 0.0
        else:
            # For reporting, calculate an average efficiency or a representative one
            # This will be the overall efficiency from the _calculate_and_apply_separation call
            pass # Calculated later in _calculate_and_apply_separation
    else: # Vertical
        if V_g_effective_separator_fps <= 0:
            st.warning("Invalid vertical separator gas velocity for gravity settling calculation. Efficiency set to 0.")
            gravity_separation_efficiency = 0.0
        else:
            # For reporting, calculate an average efficiency or a representative one
            pass # Calculated later in _calculate_and_apply_separation
    
    results['gravity_separation_efficiency'] = gravity_separation_efficiency # This will be updated after calling _calculate_and_apply_separation

    # Step 9: Mist Extractor Performance
    # Get K-deration factor based on pressure
    k_deration_factor = get_k_deration_factor(inputs['pressure_psig_input'])
    results['k_deration_factor'] = k_deration_factor

    mist_extractor_separation_efficiency = 0.0
    
    if inputs['mist_extractor_type'] == "Mesh Pad":
        mesh_pad_params = MESH_PAD_PARAMETERS[inputs['mesh_pad_type']]
        # Override thickness with user input
        mesh_pad_params_with_user_thickness = mesh_pad_params.copy()
        mesh_pad_params_with_user_thickness["thickness_in"] = inputs['mesh_pad_thickness_in']
        results['mesh_pad_params'] = mesh_pad_params_with_user_thickness # Store for reporting
        
        # The efficiency function will be called in _calculate_and_apply_separation
        # It needs rho_l_fps, rho_g_fps, mu_g_fps, V_g_effective_separator_fps, and mesh_pad_params_with_user_thickness
        pass # Efficiency calculated later
    
    elif inputs['mist_extractor_type'] == "Vane-Type":
        vane_type_params = VANE_PACK_PARAMETERS[inputs['vane_type']]
        # Override with user inputs for flow_direction, num_bends, spacing, angle
        vane_type_params_with_user_inputs = vane_type_params.copy()
        vane_type_params_with_user_inputs["flow_direction"] = inputs['vane_flow_direction']
        vane_type_params_with_user_inputs["number_of_bends"] = inputs['vane_num_bends']
        vane_type_params_with_user_inputs["vane_spacing_in"] = inputs['vane_spacing_in']
        vane_type_params_with_user_inputs["bend_angle_degree"] = inputs['vane_bend_angle_deg']
        results['vane_type_params'] = vane_type_params_with_user_inputs # Store for reporting
        
        pass # Efficiency calculated later

    elif inputs['mist_extractor_type'] == "Cyclonic":
        cyclone_type_params = CYCLONE_PARAMETERS[inputs['cyclone_type']]
        # Override with user inputs for diameter, length, swirl angle
        cyclone_type_params_with_user_inputs = cyclone_type_params.copy()
        cyclone_type_params_with_user_inputs["cyclone_inside_diameter_in"] = inputs['cyclone_diameter_in']
        cyclone_type_params_with_user_inputs["cyclone_length_in"] = inputs['cyclone_length_in']
        cyclone_type_params_with_user_inputs["inlet_swirl_angle_degree"] = inputs['cyclone_swirl_angle_deg']
        results['cyclone_type_params'] = cyclone_type_params_with_user_inputs # Store for reporting

        pass # Efficiency calculated later

    results['mist_extractor_separation_efficiency'] = mist_extractor_separation_efficiency # Updated after separation call

    return results


# Initialize session state for inputs and results if not already present
if 'inputs' not in st.session_state:
    st.session_state.inputs = {
        'D_input': 0.3048, # m (1 ft) - Feed pipe diameter
        'rho_l_input': 640.7, # kg/m3 (40 lb/ft3)
        'mu_l_input': 0.000743, # Pa.s (0.0005 lb/ft-sec)
        'V_g_input': 15.24, # m/s (50 ft/sec) - Superficial gas velocity in feed pipe
        'rho_g_input': 1.6018, # kg/m3 (0.1 lb/ft3)
        'mu_g_input': 0.00001488, # Pa.s (0.00001 lb/ft-sec)
        'sigma_custom': 0.012, # Default N/m (from user request)
        'inlet_device': "No inlet device",
        'Q_liquid_mass_flow_rate_input': 0.1, # New input: kg/s (example value)
        'num_points_distribution': 20, # Default number of points
        'separator_type': "Horizontal", # New input
        'h_g_input': 0.5, # m (for horizontal)
        'L_e_input': 2.0, # m (for horizontal)
        'D_separator_input': 1.0, # m (for vertical, or vessel diameter for horizontal)
        'L_to_ME_input': 1.0, # m (Length from Inlet Device to Mist Extractor)
        'perforated_plate_option': False, # New input
        'pressure_psig_input': 500.0, # psig (example value)
        'mist_extractor_type': "Mesh Pad", # New input
        'mesh_pad_type': "Standard mesh pad", # New input
        'mesh_pad_thickness_in': 6.0, # New input (default 6 inches)
        'vane_type': "Simple vane", # New input
        'vane_flow_direction': "Upflow", # New input
        'vane_num_bends': 5, # New input
        'vane_spacing_in': 0.75, # New input
        'vane_bend_angle_deg': 45.0, # New input
        'cyclone_type': "2.0 in. cyclones", # New input
        'cyclone_diameter_in': 2.0, # New input
        'cyclone_length_in': 10.0, # New input
        'cyclone_swirl_angle_deg': 45.0, # New input
    }
    # Initialize sigma_fps based on the new default sigma_custom
    st.session_state.inputs['sigma_fps'] = to_fps(st.session_state.inputs['sigma_custom'], "surface_tension")

if 'calculation_results' not in st.session_state:
    st.session_state.calculation_results = None
if 'plot_data_original' not in st.session_state:
    st.session_state.plot_data_original = None
if 'plot_data_adjusted' not in st.session_state:
    st.session_state.plot_data_adjusted = None
if 'plot_data_after_gravity' not in st.session_state:
    st.session_state.plot_data_after_gravity = None
if 'plot_data_after_mist_extractor' not in st.session_state:
    st.session_state.plot_data_after_mist_extractor = None
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
    pressure_unit = "psig"
    in_unit = "in" # For mist extractor dimensions

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
        st.session_state.inputs['V_g_input'] = st.number_input(f"Gas Velocity in Feed Pipe ({vel_unit})", min_value=0.01, value=st.session_state.inputs['V_g_input'], format="%.2f", key='V_g_input_widget',
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
        
        # Construct the tooltip string from SURFACE_TENSION_TABLE_DYNE_CM
        tooltip_text = "Typical Liquid Surface Tension Values (N/m):\n"
        for fluid, value_dyne_cm in SURFACE_TENSION_TABLE_DYNE_CM.items():
            value_nm = value_dyne_cm * DYNE_CM_TO_NM
            tooltip_text += f"- {fluid}: {value_nm:.3f} N/m\n"

        st.session_state.inputs['sigma_custom'] = st.number_input(
            f"Liquid Surface Tension ({surf_tens_input_unit})",
            min_value=0.0001,
            value=st.session_state.inputs['sigma_custom'],
            format="%.4f",
            key='sigma_custom_input',
            help=tooltip_text # Use the constructed tooltip text here
        )
        # Update sigma_fps based on the new default sigma_custom
        st.session_state.inputs['sigma_fps'] = to_fps(st.session_state.inputs['sigma_custom'], "surface_tension")
        
        st.info(f"**Current Liquid Surface Tension:** {st.session_state.inputs['sigma_custom']:.3f} {surf_tens_input_unit}")


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

    st.subheader("Separator Dimensions and Operation")
    st.session_state.inputs['separator_type'] = st.radio(
        "Select Separator Type",
        options=["Horizontal", "Vertical"],
        index=0 if st.session_state.inputs['separator_type'] == "Horizontal" else 1,
        key='separator_type_radio'
    )

    if st.session_state.inputs['separator_type'] == "Horizontal":
        st.session_state.inputs['h_g_input'] = st.number_input(f"Gas Space Height (h_g) ({len_unit})", min_value=0.01, value=st.session_state.inputs['h_g_input'], format="%.3f", key='h_g_input_widget',
                                    help="Vertical height of the gas phase in the horizontal separator.")
        st.session_state.inputs['L_e_input'] = st.number_input(f"Effective Separation Length (L_e) ({len_unit})", min_value=0.01, value=st.session_state.inputs['L_e_input'], format="%.3f", key='L_e_input_widget',
                                    help="Horizontal length available for gas-liquid separation in the horizontal separator.")
        st.session_state.inputs['D_separator_input'] = st.number_input(f"Horizontal Separator Vessel Diameter ({len_unit})", min_value=0.1, value=st.session_state.inputs['D_separator_input'], format="%.3f", key='D_separator_input_widget',
                                    help="Diameter of the horizontal separator vessel. Used for context, not directly in gravity settling calculations if h_g is provided.")
    else: # Vertical
        st.session_state.inputs['D_separator_input'] = st.number_input(f"Vertical Separator Diameter ({len_unit})", min_value=0.1, value=st.session_state.inputs['D_separator_input'], format="%.3f", key='D_separator_input_widget',
                                    help="Diameter of the vertical separator vessel.")
        st.session_state.inputs['L_e_input'] = st.number_input(f"Gas Gravity Section Height (L_e) ({len_unit})", min_value=0.01, value=st.session_state.inputs['L_e_input'], format="%.3f", key='L_e_input_widget',
                                    help="Vertical height of the gas gravity separation section in the vertical separator. This is used as 'h_g' for vertical settling.")
        # For vertical, h_g_input is effectively L_e_input for gravity calculations
        st.session_state.inputs['h_g_input'] = st.session_state.inputs['L_e_input']

    st.session_state.inputs['L_to_ME_input'] = st.number_input(f"Length from Inlet Device to Mist Extractor (L_to_ME) ({len_unit})", min_value=0.0, value=st.session_state.inputs['L_to_ME_input'], format="%.3f", key='L_to_ME_input_widget',
                                help="The distance from the inlet device outlet to the mist extractor. Used for F-factor calculation (L/Di).")
    st.session_state.inputs['perforated_plate_option'] = st.checkbox("Use Perforated Plate (Flow Straightening)", value=st.session_state.inputs['perforated_plate_option'], key='perforated_plate_checkbox',
                                help="Check if a perforated plate is used to improve gas velocity profile.")
    st.session_state.inputs['pressure_psig_input'] = st.number_input(f"Operating Pressure ({pressure_unit})", min_value=0.0, value=st.session_state.inputs['pressure_psig_input'], format="%.1f", key='pressure_psig_input_widget',
                                help="Operating pressure of the separator. Used for K-deration of mist extractors.")

    st.markdown("---")
    st.subheader("Mist Extractor Configuration")
    st.session_state.inputs['mist_extractor_type'] = st.selectbox(
        "Select Mist Extractor Type",
        options=["Mesh Pad", "Vane-Type", "Cyclonic"],
        index=["Mesh Pad", "Vane-Type", "Cyclonic"].index(st.session_state.inputs['mist_extractor_type']),
        key='mist_extractor_type_select'
    )

    if st.session_state.inputs['mist_extractor_type'] == "Mesh Pad":
        current_mesh_pad_index = list(MESH_PAD_PARAMETERS.keys()).index(st.session_state.inputs['mesh_pad_type'])
        st.session_state.inputs['mesh_pad_type'] = st.selectbox(
            "Mesh Pad Type",
            options=list(MESH_PAD_PARAMETERS.keys()),
            index=current_mesh_pad_index,
            key='mesh_pad_type_select'
        )
        st.session_state.inputs['mesh_pad_thickness_in'] = st.number_input(
            f"Mesh Pad Thickness ({in_unit})",
            min_value=1.0, value=st.session_state.inputs['mesh_pad_thickness_in'], format="%.2f", key='mesh_pad_thickness_in_input',
            help="Typical thickness for mesh pads is around 6 inches."
        )
    elif st.session_state.inputs['mist_extractor_type'] == "Vane-Type":
        current_vane_type_index = list(VANE_PACK_PARAMETERS.keys()).index(st.session_state.inputs['vane_type'])
        st.session_state.inputs['vane_type'] = st.selectbox(
            "Vane Type",
            options=list(VANE_PACK_PARAMETERS.keys()),
            index=current_vane_type_index,
            key='vane_type_select'
        )
        current_vane_flow_direction_index = ["Upflow", "Horizontal"].index(st.session_state.inputs['vane_flow_direction'])
        st.session_state.inputs['vane_flow_direction'] = st.selectbox(
            "Flow Direction",
            options=["Upflow", "Horizontal"],
            index=current_vane_flow_direction_index,
            key='vane_flow_direction_select'
        )
        st.session_state.inputs['vane_num_bends'] = st.number_input(
            "Number of Bends",
            min_value=1, max_value=10, value=st.session_state.inputs['vane_num_bends'], step=1, key='vane_num_bends_input',
            help="Typical range is 5-8 bends."
        )
        st.session_state.inputs['vane_spacing_in'] = st.number_input(
            f"Vane Spacing ({in_unit})",
            min_value=0.1, value=st.session_state.inputs['vane_spacing_in'], format="%.2f", key='vane_spacing_in_input',
            help="Typical range is 0.5-1 inch."
        )
        st.session_state.inputs['vane_bend_angle_deg'] = st.number_input(
            f"Bend Angle (degrees)",
            min_value=1.0, max_value=90.0, value=st.session_state.inputs['vane_bend_angle_deg'], format="%.1f", key='vane_bend_angle_deg_input',
            help="Typical range is 30-60 degrees, 45 degrees is most common."
        )
    elif st.session_state.inputs['mist_extractor_type'] == "Cyclonic":
        current_cyclone_type_index = list(CYCLONE_PARAMETERS.keys()).index(st.session_state.inputs['cyclone_type'])
        st.session_state.inputs['cyclone_type'] = st.selectbox(
            "Cyclone Type",
            options=list(CYCLONE_PARAMETERS.keys()),
            index=current_cyclone_type_index,
            key='cyclone_type_select'
        )
        st.session_state.inputs['cyclone_diameter_in'] = st.number_input(
            f"Cyclone Diameter ({in_unit})",
            min_value=0.1, value=st.session_state.inputs['cyclone_diameter_in'], format="%.2f", key='cyclone_diameter_in_input',
            help="Inside diameter of individual cyclone tube."
        )
        st.session_state.inputs['cyclone_length_in'] = st.number_input(
            f"Cyclone Length ({in_unit})",
            min_value=1.0, value=st.session_state.inputs['cyclone_length_in'], format="%.2f", key='cyclone_length_in_input',
            help="Length of individual cyclone tube."
        )
        st.session_state.inputs['cyclone_swirl_angle_deg'] = st.number_input(
            f"Inlet Swirl Angle (degrees)",
            min_value=1.0, max_value=90.0, value=st.session_state.inputs['cyclone_swirl_angle_deg'], format="%.1f", key='cyclone_swirl_angle_deg_input',
            help="Inlet swirl angle of the cyclone."
        )


    # When inputs on this page change, trigger recalculation for initial state
    import datetime
    st.session_state.report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Only perform main scalar calculations here
        st.session_state.calculation_results = _perform_main_calculations(st.session_state.inputs)
        
        # Generate initial distribution data (after inlet device, before gravity settling)
        st.session_state.plot_data_original = _generate_initial_distribution_data(
            st.session_state.calculation_results['dv50_original_fps'],
            st.session_state.calculation_results['d_max_original_fps'],
            st.session_state.inputs['num_points_distribution'],
            st.session_state.calculation_results['E_fraction'],
            st.session_state.inputs['Q_liquid_mass_flow_rate_input'],
            st.session_state.inputs['rho_l_input']
        )

        st.session_state.plot_data_adjusted = _generate_initial_distribution_data(
            st.session_state.calculation_results['dv50_adjusted_fps'],
            st.session_state.calculation_results['d_max_adjusted_fps'],
            st.session_state.inputs['num_points_distribution'],
            st.session_state.calculation_results['E_fraction'],
            st.session_state.inputs['Q_liquid_mass_flow_rate_input'],
            st.session_state.inputs['rho_l_input']
        )

        # Calculate and apply gravity settling
        if st.session_state.inputs['separator_type'] == "Horizontal":
            st.session_state.plot_data_after_gravity = _calculate_and_apply_separation(
                st.session_state.plot_data_adjusted, # Input is the adjusted distribution
                separation_stage_efficiency_func=gravity_efficiency_func_horizontal,
                is_gravity_stage=True,
                V_g_eff_sep_fps=st.session_state.calculation_results['V_g_effective_separator_fps'],
                h_g_sep_fps=to_fps(st.session_state.inputs['h_g_input'], 'length'),
                L_e_sep_fps=to_fps(st.session_state.inputs['L_e_input'], 'length'),
                rho_l_fps=to_fps(st.session_state.inputs['rho_l_input'], 'density'),
                rho_g_fps=to_fps(st.session_state.inputs['rho_g_input'], 'density'),
                mu_g_fps=to_fps(st.session_state.inputs['mu_g_input'], 'viscosity'),
                separator_type=st.session_state.inputs['separator_type']
            )
        else: # Vertical
            st.session_state.plot_data_after_gravity = _calculate_and_apply_separation(
                st.session_state.plot_data_adjusted, # Input is the adjusted distribution
                separation_stage_efficiency_func=gravity_efficiency_func_vertical,
                is_gravity_stage=True,
                V_g_eff_sep_fps=st.session_state.calculation_results['V_g_effective_separator_fps'],
                # For vertical, h_g_input is effectively L_e_input for gravity calculations
                h_g_sep_fps=to_fps(st.session_state.inputs['L_e_input'], 'length'), 
                L_e_sep_fps=to_fps(st.session_state.inputs['L_e_input'], 'length'), # Pass L_e_input for vertical time_settle calc
                rho_l_fps=to_fps(st.session_state.inputs['rho_l_input'], 'density'),
                rho_g_fps=to_fps(st.session_state.inputs['rho_g_input'], 'density'),
                mu_g_fps=to_fps(st.session_state.inputs['mu_g_input'], 'viscosity'),
                separator_type=st.session_state.inputs['separator_type']
            )
        
        # Update the overall gravity separation efficiency in results for reporting
        if st.session_state.plot_data_after_gravity:
            st.session_state.calculation_results['gravity_separation_efficiency'] = st.session_state.plot_data_after_gravity['overall_separation_efficiency']
        else:
            st.session_state.calculation_results['gravity_separation_efficiency'] = 0.0

        # Calculate and apply mist extractor efficiency
        if st.session_state.plot_data_after_gravity and st.session_state.plot_data_after_gravity['dp_values_ft'].size > 0:
            if st.session_state.inputs['mist_extractor_type'] == "Mesh Pad":
                mesh_pad_params = MESH_PAD_PARAMETERS[st.session_state.inputs['mesh_pad_type']]
                mesh_pad_params_with_user_thickness = mesh_pad_params.copy()
                mesh_pad_params_with_user_thickness["thickness_in"] = st.session_state.inputs['mesh_pad_thickness_in']
                
                st.session_state.plot_data_after_mist_extractor = _calculate_and_apply_separation(
                    st.session_state.plot_data_after_gravity,
                    separation_stage_efficiency_func=mesh_pad_efficiency_func,
                    V_g_eff_sep_fps=st.session_state.calculation_results['V_g_effective_separator_fps'],
                    rho_l_fps=to_fps(st.session_state.inputs['rho_l_input'], 'density'),
                    rho_g_fps=to_fps(st.session_state.inputs['rho_g_input'], 'density'),
                    mu_g_fps=to_fps(st.session_state.inputs['mu_g_input'], 'viscosity'),
                    mist_extractor_type_str="Mesh Pad", # Pass the type string
                    mesh_pad_type_params_fps=mesh_pad_params_with_user_thickness
                )
            elif st.session_state.inputs['mist_extractor_type'] == "Vane-Type":
                vane_type_params = VANE_PACK_PARAMETERS[st.session_state.inputs['vane_type']]
                vane_type_params_with_user_inputs = vane_type_params.copy()
                vane_type_params_with_user_inputs["flow_direction"] = st.session_state.inputs['vane_flow_direction']
                vane_type_params_with_user_inputs["number_of_bends"] = st.session_state.inputs['vane_num_bends']
                vane_type_params_with_user_inputs["vane_spacing_in"] = st.session_state.inputs['vane_spacing_in']
                vane_type_params_with_user_inputs["bend_angle_degree"] = st.session_state.inputs['vane_bend_angle_deg']

                st.session_state.plot_data_after_mist_extractor = _calculate_and_apply_separation(
                    st.session_state.plot_data_after_gravity,
                    separation_stage_efficiency_func=vane_type_efficiency_func,
                    V_g_eff_sep_fps=st.session_state.calculation_results['V_g_effective_separator_fps'],
                    rho_l_fps=to_fps(st.session_state.inputs['rho_l_input'], 'density'),
                    rho_g_fps=to_fps(st.session_state.inputs['rho_g_input'], 'density'),
                    mu_g_fps=to_fps(st.session_state.inputs['mu_g_input'], 'viscosity'),
                    mist_extractor_type_str="Vane-Type", # Pass the type string
                    vane_type_params_fps=vane_type_params_with_user_inputs
                )
            elif st.session_state.inputs['mist_extractor_type'] == "Cyclonic":
                cyclone_type_params = CYCLONE_PARAMETERS[st.session_state.inputs['cyclone_type']]
                cyclone_type_params_with_user_inputs = cyclone_type_params.copy()
                cyclone_type_params_with_user_inputs["cyclone_inside_diameter_in"] = st.session_state.inputs['cyclone_diameter_in']
                cyclone_type_params_with_user_inputs["cyclone_length_in"] = st.session_state.inputs['cyclone_length_in']
                cyclone_type_params_with_user_inputs["inlet_swirl_angle_degree"] = st.session_state.inputs['cyclone_swirl_angle_deg']

                st.session_state.plot_data_after_mist_extractor = _calculate_and_apply_separation(
                    st.session_state.plot_data_after_gravity,
                    separation_stage_efficiency_func=demisting_cyclone_efficiency_func,
                    V_g_eff_sep_fps=st.session_state.calculation_results['V_g_effective_separator_fps'],
                    rho_l_fps=to_fps(st.session_state.inputs['rho_l_input'], 'density'),
                    rho_g_fps=to_fps(st.session_state.inputs['rho_g_input'], 'density'),
                    mu_g_fps=to_fps(st.session_state.inputs['mu_g_input'], 'viscosity'),
                    mist_extractor_type_str="Cyclonic", # Pass the type string
                    cyclone_type_params_fps=cyclone_type_params_with_user_inputs
                )
            else:
                st.session_state.plot_data_after_mist_extractor = st.session_state.plot_data_after_gravity # No mist extractor selected, so no change
            
            # Update the overall mist extractor separation efficiency in results for reporting
            if st.session_state.plot_data_after_mist_extractor:
                st.session_state.calculation_results['mist_extractor_separation_efficiency'] = st.session_state.plot_data_after_mist_extractor['overall_separation_efficiency']
            else:
                st.session_state.calculation_results['mist_extractor_separation_efficiency'] = 0.0

        else:
            st.session_state.plot_data_after_mist_extractor = None


    except Exception as e:
        st.error(f"An error occurred during calculation: {e}")
        st.session_state.calculation_results = None
        st.session_state.plot_data_original = None
        st.session_state.plot_data_adjusted = None
        st.session_state.plot_data_after_gravity = None
        st.session_state.plot_data_after_mist_extractor = None


# --- Page: Calculation Steps ---
elif page == "Calculation Steps":
    st.header("2. Step-by-Step Calculation Results")

    if st.session_state.calculation_results:
        results = st.session_state.calculation_results
        inputs = st.session_state.inputs
        
        # Define unit labels for SI system
        len_unit = "m"
        dens_unit = "kg/m³"
        vel_unit = "m/s"
        visc_unit = "Pa·s"
        momentum_unit = "Pa"
        micron_unit_label = "µm"
        mass_flow_unit = "kg/s"
        vol_flow_unit = "m³/s" # New unit for Streamlit display
        pressure_unit = "psig"
        in_unit = "in"

        # Display inputs used for calculation (original SI values)
        st.subheader("Inputs Used for Calculation (SI Units)")
        st.write(f"Pipe Inside Diameter (D): {inputs['D_input']:.4f} {len_unit}")
        st.write(f"Liquid Density (ρl): {inputs['rho_l_input']:.2f} {dens_unit}")
        st.write(f"Liquid Viscosity (μl): {inputs['mu_l_input']:.8f} {visc_unit}")
        st.write(f"Gas Velocity in Feed Pipe (Vg): {inputs['V_g_input']:.2f} {vel_unit}")
        st.write(f"Gas Density (ρg): {inputs['rho_g_input']:.5f} {dens_unit}")
        st.write(f"Gas Viscosity (μg): {inputs['mu_g_input']:.9f} {visc_unit}")
        # Display selected surface tension in SI units
        sigma_display_val = inputs['sigma_custom'] # Use sigma_custom for display
        st.write(f"Liquid Surface Tension (σ): {sigma_display_val:.3f} N/m")
        st.write(f"Selected Inlet Device: {inputs['inlet_device']}")
        st.write(f"Total Liquid Mass Flow Rate: {inputs['Q_liquid_mass_flow_rate_input']:.2f} {mass_flow_unit}") # New input
        st.write(f"Operating Pressure: {inputs['pressure_psig_input']:.1f} {pressure_unit}")
        st.write(f"Separator Type: {inputs['separator_type']}")
        if inputs['separator_type'] == "Horizontal":
            st.write(f"Gas Space Height (hg): {inputs['h_g_input']:.3f} {len_unit}")
            st.write(f"Effective Separation Length (Le): {inputs['L_e_input']:.3f} {len_unit}")
        else: # Vertical
            st.write(f"Separator Diameter: {inputs['D_separator_input']:.3f} {len_unit}")
            st.write(f"Gas Gravity Section Height (L_e): {inputs['L_e_input']:.3f} {len_unit}")
        st.write(f"Length from Inlet Device to Mist Extractor (L_to_ME): {inputs['L_to_ME_input']:.3f} {len_unit}")
        st.write(f"Perforated Plate Used: {'Yes' if inputs['perforated_plate_option'] else 'No'}")
        st.write(f"Mist Extractor Type: {inputs['mist_extractor_type']}")
        if inputs['mist_extractor_type'] == "Mesh Pad":
            st.write(f"  Mesh Pad Type: {inputs['mesh_pad_type']}")
            st.write(f"  Mesh Pad Thickness: {inputs['mesh_pad_thickness_in']:.2f} {in_unit}")
        elif inputs['mist_extractor_type'] == "Vane-Type":
            st.write(f"  Vane Type: {inputs['vane_type']}")
            st.write(f"  Flow Direction: {inputs['vane_flow_direction']}")
            st.write(f"  Number of Bends: {inputs['vane_num_bends']}")
            st.write(f"  Vane Spacing: {inputs['vane_spacing_in']:.2f} {in_unit}")
            st.write(f"  Bend Angle: {inputs['vane_bend_angle_deg']:.1f} deg")
        elif inputs['mist_extractor_type'] == "Cyclonic":
            st.write(f"  Cyclone Type: {inputs['cyclone_type']}")
            st.write(f"  Cyclone Diameter: {inputs['cyclone_diameter_in']:.2f} {in_unit}")
            st.write(f"  Cyclone Length: {inputs['cyclone_length_in']:.2f} {in_unit}")
            st.write(f"  Inlet Swirl Angle: {inputs['cyclone_swirl_angle_deg']:.1f} deg")
        st.markdown("---")

        # Step 1: Calculate Superficial Gas Reynolds Number (Re_g)
        st.markdown("#### Step 1: Calculate Superficial Gas Reynolds Number ($Re_g$)")
        D_pipe_fps = to_fps(inputs['D_input'], "length")
        V_g_input_fps = to_fps(inputs['V_g_input'], "velocity")
        rho_g_fps = to_fps(inputs['rho_g_input'], "density")
        mu_g_fps = to_fps(inputs['mu_g_input'], "viscosity")

        st.write(f"Equation: $Re_g = \\frac{{D \\cdot V_g \\cdot \\rho_g}}{{\\mu_g}}$")
        st.write(f"Calculation (FPS): $Re_g = \\frac{{{D_pipe_fps:.2f} \\text{{ ft}} \\cdot {V_g_input_fps:.2f} \\text{{ ft/sec}} \\cdot {rho_g_fps:.4f} \\text{{ lb/ft}}^3}}{{{mu_g_fps:.8f} \\text{{ lb/ft-sec}}}} = {results['Re_g']:.2f}$")
        st.success(f"**Result:** Superficial Gas Reynolds Number ($Re_g$) = **{results['Re_g']:.2f}** (dimensionless)")

        st.markdown("---")

        # Step 2: Calculate Volume Median Diameter ($d_{v50}$) without inlet device effect
        st.markdown("#### Step 2: Calculate Initial Volume Median Diameter ($d_{v50}$) (Kataoka et al., 1983)")
        rho_l_fps = to_fps(inputs['rho_l_input'], "density")
        mu_l_fps = to_fps(inputs['mu_l_input'], "viscosity")

        dv50_original_display = from_fps(results['dv50_original_fps'], "length")
        
        # Updated LaTeX formula for display
        st.write(f"Equation: $d_{{v50}} = 0.01 \\left(\\frac{{\\sigma}}{{\\rho_g V_g^2}}\\right) Re_g^{{2/3}} \\left(\\frac{{\\rho_g}}{{\\rho_l}}\\right)^{{-1/3}} \\left(\\frac{{\\mu_g}}{{\\mu_l}}\\right)^{{2/3}}$")
        st.write(f"Calculation (FPS): $d_{{v50}} = 0.01 \\left(\\frac{{{inputs['sigma_fps']:.4f}}}{{{rho_g_fps:.4f} \\cdot ({V_g_input_fps:.2f})^2}}\\right) ({results['Re_g']:.2f})^{{2/3}} \\left(\\frac{{{rho_g_fps:.4f}}}{{{rho_l_fps:.2f}}}\\right)^{{-0.333}} \\left(\\frac{{{mu_g_fps:.8f}}}{{{mu_l_fps:.7f}}}\\right)^{{0.667}}$")
        st.success(f"**Result:** Initial Volume Median Diameter ($d_{{v50}}$) = **{results['dv50_original_fps'] * FT_TO_MICRON:.2f} {micron_unit_label}** ({dv50_original_display:.6f} {len_unit})")

        st.markdown("---")

        # Step 3: Determine Inlet Momentum (rho_g V_g^2)
        st.markdown("#### Step 3: Calculate Inlet Momentum ($\\rho_g V_g^2$)")
        rho_v_squared_display = from_fps(results['rho_v_squared_fps'], "momentum")
        st.write(f"Equation: $\\rho_g V_g^2 = \\rho_g \\cdot V_g^2$")
        st.write(f"Calculation (FPS): $\\rho_g V_g^2 = {rho_g_fps:.4f} \\text{{ lb/ft}}^3 \\cdot ({V_g_input_fps:.2f} \\text{{ ft/sec}})^2 = {results['rho_v_squared_fps']:.2f} \\text{{ lb/ft-sec}}^2$")
        st.success(f"**Result:** Inlet Momentum ($\\rho_g V_g^2$) = **{rho_v_squared_display:.2f} {momentum_unit}**")

        st.markdown("---")

        # Step 4: Apply Inlet Device "Droplet Size Distribution Shift Factor"
        st.markdown("#### Step 4: Apply Inlet Device Effect (Droplet Size Distribution Shift Factor)")
        st.write(f"Selected Inlet Device: **{inputs['inlet_device']}**")
        dv50_adjusted_display = from_fps(results['dv50_adjusted_fps'], "length")
        st.write(f"Based on Figure 9 from the article, for an inlet momentum of {rho_v_squared_display:.2f} {momentum_unit} and a '{inputs['inlet_device']}' device, the estimated shift factor is **{results['shift_factor']:.3f}**.")
        st.write(f"Equation: $d_{{v50, adjusted}} = d_{{v50, original}} \\cdot \\text{{Shift Factor}}$")
        st.write(f"Calculation (FPS): $d_{{v50, adjusted}} = {results['dv50_original_fps']:.6f} \\text{{ ft}} \\cdot {results['shift_factor']:.3f} = {results['dv50_adjusted_fps']:.6f} \\text{{ ft}}$")
        st.success(f"**Result:** Adjusted Volume Median Diameter ($d_{{v50}}$) = **{results['dv50_adjusted_fps'] * FT_TO_MICRON:.2f} {micron_unit_label}** ({dv50_adjusted_display:.6f} {len_unit})")

        st.markdown("---")

        # Step 5: Calculate parameters for Upper-Limit Log Normal Distribution
        st.markdown("#### Step 5: Calculate Parameters for Upper-Limit Log Normal Distribution")
        d_max_original_display = from_fps(results['d_max_original_fps'], "length")
        d_max_adjusted_display = from_fps(results['d_max_adjusted_fps'], "length")
        st.write(f"Using typical values from the article: $a = {A_DISTRIBUTION}$ and $\\delta = {DELTA_DISTRIBUTION}$.")
        st.write(f"For **Original** $d_{{v50}}$:")
        st.write(f"Equation: $d_{{max, original}} = a \\cdot d_{{v50, original}}$")
        st.write(f"Calculation (FPS): $d_{{max, original}} = {A_DISTRIBUTION} \\cdot {results['dv50_original_fps']:.6f} \\text{{ ft}} = {results['d_max_original_fps']:.6f} \\text{{ ft}}$")
        st.success(f"**Result:** Maximum Droplet Size (Original $d_{{max}}$) = **{results['d_max_original_fps'] * FT_TO_MICRON:.2f} {micron_unit_label}** ({d_max_original_display:.6f} {len_unit})")
        st.write(f"For **Adjusted** $d_{{v50}}$:")
        st.write(f"Equation: $d_{{max, adjusted}} = a \\cdot d_{{v50, adjusted}}$")
        st.write(f"Calculation (FPS): $d_{{max, adjusted}} = {A_DISTRIBUTION} \\cdot {results['dv50_adjusted_fps']:.6f} \\text{{ ft}} = {results['d_max_adjusted_fps']:.6f} \\text{{ ft}}$")
        st.success(f"**Result:** Maximum Droplet Size (Adjusted $d_{{max}}$) = **{results['d_max_adjusted_fps'] * FT_TO_MICRON:.2f} {micron_unit_label}** ({d_max_adjusted_display:.6f} {len_unit})")

        st.markdown("---")

        # Step 6: Entrainment Fraction (E) Calculation
        st.markdown("#### Step 6: Calculate Entrainment Fraction (E)")
        st.write(f"Gas Velocity (Ug): {inputs['V_g_input']:.2f} {vel_unit}")
        st.write(f"Liquid Loading (Wl): {inputs['Q_liquid_mass_flow_rate_input']:.2f} {mass_flow_unit}")
        st.success(f"**Result:** Entrainment Fraction (E) = **{results['E_fraction']:.4f}** (dimensionless)")
        st.success(f"**Result:** Total Entrained Liquid Mass Flow Rate = **{results['Q_entrained_total_mass_flow_rate_si']:.4f} {mass_flow_unit}**")
        st.success(f"**Result:** Total Entrained Liquid Volume Flow Rate = **{results['Q_entrained_total_volume_flow_rate_si']:.6f} {vol_flow_unit}**") # New total volume flow
        st.markdown("---")

        # Step 7: Calculate F-factor and Effective Gas Velocity in Separator
        st.markdown("#### Step 7: Calculate F-factor and Effective Gas Velocity in Separator")
        D_pipe_fps = to_fps(inputs['D_input'], "length")
        L_to_ME_fps = to_fps(inputs['L_to_ME_input'], 'length')
        L_over_Di = L_to_ME_fps / D_pipe_fps
        st.write(f"L/Di Ratio (Length from Inlet Device to Mist Extractor / Pipe Inside Diameter): {L_to_ME_fps:.2f} ft / {D_pipe_fps:.2f} ft = {L_over_Di:.2f}")
        st.write(f"Inlet Device: {inputs['inlet_device']}")
        st.write(f"Perforated Plate Used: {'Yes' if inputs['perforated_plate_option'] else 'No'}")
        st.write(f"Calculated F-factor (from Fig. 2): {results['F_factor']:.3f}")

        V_g_effective_separator_display = from_fps(results['V_g_effective_separator_fps'], 'velocity')
        if inputs['separator_type'] == "Vertical":
            D_separator_fps = to_fps(inputs['D_separator_input'], 'length')
            A_pipe_fps = np.pi * (D_pipe_fps / 2)**2
            A_separator_gas_vertical_fps = np.pi * (D_separator_fps / 2)**2
            V_g_superficial_separator_fps = (to_fps(inputs['V_g_input'], 'velocity') * A_pipe_fps) / A_separator_gas_vertical_fps
            st.write(f"Superficial Gas Velocity in Vertical Separator: {from_fps(V_g_superficial_separator_fps, 'velocity'):.2f} {vel_unit}")
            st.write(f"Equation: $V_{{g,effective}} = V_{{g,superficial}} / F$")
            st.write(f"Calculation (FPS): $V_{{g,effective}} = {V_g_superficial_separator_fps:.2f} \\text{{ ft/sec}} / {results['F_factor']:.3f} = {results['V_g_effective_separator_fps']:.2f} \\text{{ ft/sec}}$")
        else: # Horizontal
            st.write(f"Equation: $V_{{g,effective}} = V_{{g,input}} / F$ (assuming input Vg is superficial in separator gas section)")
            st.write(f"Calculation (FPS): $V_{{g,effective}} = {to_fps(inputs['V_g_input'], 'velocity'):.2f} \\text{{ ft/sec}} / {results['F_factor']:.3f} = {results['V_g_effective_separator_fps']:.2f} \\text{{ ft/sec}}$")
        st.success(f"**Result:** Effective Gas Velocity in Separator ($V_{{g,effective}}$) = **{V_g_effective_separator_display:.2f} {vel_unit}**")
        st.markdown("---")

        # Step 8: Gas Gravity Separation Section Efficiency
        st.markdown("#### Step 8: Gas Gravity Separation Section Efficiency")
        st.write(f"Separator Type: **{inputs['separator_type']}**")
        if inputs['separator_type'] == "Horizontal":
            st.write(f"Gas Space Height (h_g): {inputs['h_g_input']:.3f} {len_unit}")
            st.write(f"Effective Separation Length (L_e): {inputs['L_e_input']:.3f} {len_unit}")
            st.write("For each droplet size, the separation efficiency is calculated based on its terminal velocity and the available settling time/distance.")
        else: # Vertical
            st.write(f"Separator Diameter: {inputs['D_separator_input']:.3f} {len_unit}")
            st.write(f"Gas Gravity Section Height (L_e): {inputs['L_e_input']:.3f} {len_unit}")
            st.write("For a vertical separator, a droplet is separated if its terminal settling velocity is greater than the effective upward gas velocity.")
        
        st.success(f"**Result:** Overall Separation Efficiency of Gas Gravity Section = **{results['gravity_separation_efficiency']:.2%}**")
        if st.session_state.plot_data_after_gravity:
            st.success(f"**Result:** Total Entrained Liquid Mass Flow Rate After Gravity Settling = **{st.session_state.plot_data_after_gravity['total_entrained_mass_flow_rate_si']:.4f} {mass_flow_unit}**")
            st.success(f"**Result:** Total Entrained Liquid Volume Flow Rate After Gravity Settling = **{st.session_state.plot_data_after_gravity['total_entrained_volume_flow_rate_si']:.6f} {vol_flow_unit}**")
        else:
            st.warning("Gravity settling results not available. Please check inputs and previous steps.")

        # Display detailed table for gravity separation
        if st.session_state.plot_data_after_gravity and st.session_state.plot_data_after_gravity['gravity_details_table_data']:
            st.markdown("##### Detailed Droplet Separation Performance in Gas Gravity Section")
            gravity_table_df = pd.DataFrame(st.session_state.plot_data_after_gravity['gravity_details_table_data'])
            
            # Format columns for display
            st.dataframe(gravity_table_df.style.format({
                "dp_microns": "{:.2f}",
                "Vt_ftps": "{:.4f}",
                "Cd": "{:.4f}",
                "Re_p": "{:.2e}", # Scientific notation for Reynolds number
                "Time Settle (s)": "{:.4f}",
                "h_max_settle (ft)": "{:.4f}",
                "Edp": "{:.2%}" # Percentage for efficiency
            }))
        else:
            st.info("Detailed droplet separation data for gravity section not available.")


        st.markdown("---")

        # Step 9: Mist Extractor Performance
        st.markdown("#### Step 9: Mist Extractor Performance")
        st.write(f"Mist Extractor Type: **{inputs['mist_extractor_type']}**")
        st.write(f"Operating Pressure: {inputs['pressure_psig_input']:.1f} {pressure_unit}")
        st.write(f"K-Deration Factor (from Table 3): {results['k_deration_factor']:.3f}")

        if inputs['mist_extractor_type'] == "Mesh Pad":
            mesh_pad_params_fps = results['mesh_pad_params']
            st.write(f"  Mesh Pad Type: {inputs['mesh_pad_type']}")
            st.write(f"  Mesh Pad Thickness: {inputs['mesh_pad_thickness_in']:.2f} {in_unit}")
            st.write(f"  Wire Diameter: {mesh_pad_params_fps['wire_diameter_in']:.3f} {in_unit}")
            st.write(f"  Specific Surface Area: {mesh_pad_params_fps['specific_surface_area_ft2_ft3']:.1f} ft²/ft³")
            st.write(f"  Base K_s: {mesh_pad_params_fps['Ks_ft_sec']:.2f} ft/sec")
            st.write(f"  Liquid Load Capacity: {mesh_pad_params_fps['liquid_load_gal_min_ft2']:.2f} gal/min/ft²")
            st.write("  Efficiency calculated using Stokes' number, single-wire efficiency (Fig. 8), and mesh-pad removal efficiency (Eq. 14).")

        elif inputs['mist_extractor_type'] == "Vane-Type":
            vane_type_params_fps = results['vane_type_params']
            st.write(f"  Vane Type: {inputs['vane_type']}")
            st.write(f"  Flow Direction: {inputs['vane_flow_direction']}")
            st.write(f"  Number of Bends: {inputs['vane_num_bends']}")
            st.write(f"  Vane Spacing: {inputs['vane_spacing_in']:.2f} {in_unit}")
            st.write(f"  Bend Angle: {inputs['vane_bend_angle_deg']:.1f} deg")
            st.write(f"  Base K_s (Upflow): {vane_type_params_fps['Ks_ft_sec_upflow']:.2f} ft/sec")
            st.write(f"  Base K_s (Horizontal): {vane_type_params_fps['Ks_ft_sec_horizontal']:.2f} ft/sec")
            st.write(f"  Liquid Load Capacity: {vane_type_params_fps['liquid_load_gal_min_ft2']:.2f} gal/min/ft²")
            st.write("  Efficiency calculated using Eq. 15.")

        elif inputs['mist_extractor_type'] == "Cyclonic":
            cyclone_type_params_fps = results['cyclone_type_params']
            st.write(f"  Cyclone Type: {inputs['cyclone_type']}")
            st.write(f"  Cyclone Diameter: {inputs['cyclone_diameter_in']:.2f} {in_unit}")
            st.write(f"  Cyclone Length: {inputs['cyclone_length_in']:.2f} {in_unit}")
            st.write(f"  Inlet Swirl Angle: {inputs['cyclone_swirl_angle_deg']:.1f} deg")
            st.write(f"  Base K_s: {cyclone_type_params_fps['Ks_ft_sec_bundle_face_area']:.2f} ft/sec")
            st.write(f"  Liquid Load Capacity: {cyclone_type_params_fps['liquid_load_gal_min_ft2_bundle_face_area']:.2f} gal/min/ft²")
            st.write("  Efficiency calculated using Eq. 16.")

        st.success(f"**Result:** Overall Separation Efficiency of Mist Extractor = **{results['mist_extractor_separation_efficiency']:.2%}**")
        if st.session_state.plot_data_after_mist_extractor:
            st.success(f"**Result:** Total Entrained Liquid Mass Flow Rate After Mist Extractor = **{st.session_state.plot_data_after_mist_extractor['total_entrained_mass_flow_rate_si']:.4f} {mass_flow_unit}**")
            st.success(f"**Result:** Total Entrained Liquid Volume Flow Rate After Mist Extractor = **{st.session_state.plot_data_after_mist_extractor['total_entrained_volume_flow_rate_si']:.6f} {vol_flow_unit}**")
        else:
            st.warning("Mist extractor results not available. Please check inputs and previous steps.")

        # Display detailed table for mist extractor performance
        if st.session_state.plot_data_after_mist_extractor and st.session_state.plot_data_after_mist_extractor['mist_extractor_details_table_data']:
            st.markdown("##### Detailed Droplet Separation Performance in Mist Extractor")
            mist_extractor_table_df = pd.DataFrame(st.session_state.plot_data_after_mist_extractor['mist_extractor_details_table_data'])
            
            # Format columns based on mist extractor type
            if inputs['mist_extractor_type'] == "Mesh Pad":
                st.dataframe(mist_extractor_table_df.style.format({
                    "dp_microns": "{:.2f}",
                    "Stk": "{:.2e}",
                    "Ew": "{:.4f}",
                    "E_pad": "{:.2%}"
                }))
            elif inputs['mist_extractor_type'] == "Vane-Type":
                st.dataframe(mist_extractor_table_df.style.format({
                    "dp_microns": "{:.2f}",
                    "E_vane": "{:.2%}"
                }))
            elif inputs['mist_extractor_type'] == "Cyclonic":
                st.dataframe(mist_extractor_table_df.style.format({
                    "dp_microns": "{:.2f}",
                    "Stk_cycl": "{:.2e}",
                    "E_cycl": "{:.2%}"
                }))
        else:
            st.info("Detailed droplet separation data for mist extractor not available.")

        st.markdown("---")
        st.subheader("Final Carry-Over from Separator Outlet")
        if st.session_state.plot_data_after_mist_extractor:
            st.success(f"**Total Carry-Over Mass Flow Rate:** **{st.session_state.plot_data_after_mist_extractor['total_entrained_mass_flow_rate_si']:.4f} {mass_flow_unit}**")
            st.success(f"**Total Carry-Over Volume Flow Rate:** **{st.session_state.plot_data_after_mist_extractor['total_entrained_volume_flow_rate_si']:.6f} {vol_flow_unit}**")
        else:
            st.warning("Final carry-over results not available. Please ensure all previous steps are calculated.")

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
    # Ensure calculation_results are available before proceeding
    if st.session_state.calculation_results:
        try:
            results = st.session_state.calculation_results
            inputs = st.session_state.inputs
            num_points = inputs['num_points_distribution']
            
            # Ensure required inputs for distribution generation are available
            if 'Q_liquid_mass_flow_rate_input' in inputs and \
               inputs['Q_liquid_mass_flow_rate_input'] is not None and \
               'rho_l_input' in inputs and \
               inputs['rho_l_input'] is not None:

                # Generate initial distribution (after inlet device, before gravity settling)
                st.session_state.plot_data_original = _generate_initial_distribution_data(
                    results['dv50_original_fps'],
                    results['d_max_original_fps'],
                    num_points,
                    results['E_fraction'],
                    inputs['Q_liquid_mass_flow_rate_input'],
                    inputs['rho_l_input']
                )

                st.session_state.plot_data_adjusted = _generate_initial_distribution_data(
                    results['dv50_adjusted_fps'],
                    results.get('d_max_adjusted_fps', results['d_max_original_fps']), # Use original if adjusted not present
                    num_points,
                    results['E_fraction'],
                    inputs['Q_liquid_mass_flow_rate_input'],
                    inputs['rho_l_input']
                )

                # Calculate and apply gravity settling
                if inputs['separator_type'] == "Horizontal":
                    st.session_state.plot_data_after_gravity = _calculate_and_apply_separation(
                        st.session_state.plot_data_adjusted, # Input is the adjusted distribution
                        separation_stage_efficiency_func=gravity_efficiency_func_horizontal,
                        is_gravity_stage=True,
                        V_g_eff_sep_fps=st.session_state.calculation_results['V_g_effective_separator_fps'],
                        h_g_sep_fps=to_fps(inputs['h_g_input'], 'length'),
                        L_e_sep_fps=to_fps(inputs['L_e_input'], 'length'),
                        rho_l_fps=to_fps(inputs['rho_l_input'], 'density'),
                        rho_g_fps=to_fps(inputs['rho_g_input'], 'density'),
                        mu_g_fps=to_fps(inputs['mu_g_input'], 'viscosity'),
                        separator_type=inputs['separator_type']
                    )
                else: # Vertical
                    st.session_state.plot_data_after_gravity = _calculate_and_apply_separation(
                        st.session_state.plot_data_adjusted, # Input is the adjusted distribution
                        separation_stage_efficiency_func=gravity_efficiency_func_vertical,
                        is_gravity_stage=True,
                        V_g_eff_sep_fps=st.session_state.calculation_results['V_g_effective_separator_fps'],
                        # For vertical, h_g_input is effectively L_e_input for gravity calculations
                        h_g_sep_fps=to_fps(inputs['L_e_input'], 'length'), 
                        L_e_sep_fps=to_fps(inputs['L_e_input'], 'length'), # Pass L_e_input for vertical time_settle calc
                        rho_l_fps=to_fps(inputs['rho_l_input'], 'density'),
                        rho_g_fps=to_fps(inputs['rho_g_input'], 'density'),
                        mu_g_fps=to_fps(inputs['mu_g_input'], 'viscosity'),
                        separator_type=inputs['separator_type']
                    )
                
                # Update the overall gravity separation efficiency in results for reporting
                if st.session_state.plot_data_after_gravity:
                    st.session_state.calculation_results['gravity_separation_efficiency'] = st.session_state.plot_data_after_gravity['overall_separation_efficiency']
                else:
                    st.session_state.calculation_results['gravity_separation_efficiency'] = 0.0

                # Calculate and apply mist extractor efficiency
                if st.session_state.plot_data_after_gravity and st.session_state.plot_data_after_gravity['dp_values_ft'].size > 0:
                    if st.session_state.inputs['mist_extractor_type'] == "Mesh Pad":
                        mesh_pad_params = MESH_PAD_PARAMETERS[st.session_state.inputs['mesh_pad_type']]
                        mesh_pad_params_with_user_thickness = mesh_pad_params.copy()
                        mesh_pad_params_with_user_thickness["thickness_in"] = st.session_state.inputs['mesh_pad_thickness_in']
                        
                        st.session_state.plot_data_after_mist_extractor = _calculate_and_apply_separation(
                            st.session_state.plot_data_after_gravity,
                            separation_stage_efficiency_func=mesh_pad_efficiency_func,
                            V_g_eff_sep_fps=st.session_state.calculation_results['V_g_effective_separator_fps'],
                            rho_l_fps=to_fps(inputs['rho_l_input'], 'density'),
                            rho_g_fps=to_fps(inputs['rho_g_input'], 'density'),
                            mu_g_fps=to_fps(inputs['mu_g_input'], 'viscosity'),
                            mist_extractor_type_str="Mesh Pad", # Pass the type string
                            mesh_pad_type_params_fps=mesh_pad_params_with_user_thickness
                        )
                    elif st.session_state.inputs['mist_extractor_type'] == "Vane-Type":
                        vane_type_params = VANE_PACK_PARAMETERS[st.session_state.inputs['vane_type']]
                        vane_type_params_with_user_inputs = vane_type_params.copy()
                        vane_type_params_with_user_inputs["flow_direction"] = st.session_state.inputs['vane_flow_direction']
                        vane_type_params_with_user_inputs["number_of_bends"] = st.session_state.inputs['vane_num_bends']
                        vane_type_params_with_user_inputs["vane_spacing_in"] = st.session_state.inputs['vane_spacing_in']
                        vane_type_params_with_user_inputs["bend_angle_degree"] = st.session_state.inputs['vane_bend_angle_deg']

                        st.session_state.plot_data_after_mist_extractor = _calculate_and_apply_separation(
                            st.session_state.plot_data_after_gravity,
                            separation_stage_efficiency_func=vane_type_efficiency_func,
                            V_g_eff_sep_fps=st.session_state.calculation_results['V_g_effective_separator_fps'],
                            rho_l_fps=to_fps(inputs['rho_l_input'], 'density'),
                            rho_g_fps=to_fps(inputs['rho_g_input'], 'density'),
                            mu_g_fps=to_fps(inputs['mu_g_input'], 'viscosity'),
                            mist_extractor_type_str="Vane-Type", # Pass the type string
                            vane_type_params_fps=vane_type_params_with_user_inputs
                        )
                    elif st.session_state.inputs['mist_extractor_type'] == "Cyclonic":
                        cyclone_type_params = CYCLONE_PARAMETERS[st.session_state.inputs['cyclone_type']]
                        cyclone_type_params_with_user_inputs = cyclone_type_params.copy()
                        cyclone_type_params_with_user_inputs["cyclone_inside_diameter_in"] = st.session_state.inputs['cyclone_diameter_in']
                        cyclone_type_params_with_user_inputs["cyclone_length_in"] = st.session_state.inputs['cyclone_length_in']
                        cyclone_type_params_with_user_inputs["inlet_swirl_angle_deg"] = st.session_state.inputs['cyclone_swirl_angle_deg']

                        st.session_state.plot_data_after_mist_extractor = _calculate_and_apply_separation(
                            st.session_state.plot_data_after_gravity,
                            separation_stage_efficiency_func=demisting_cyclone_efficiency_func,
                            V_g_eff_sep_fps=st.session_state.calculation_results['V_g_effective_separator_fps'],
                            rho_l_fps=to_fps(inputs['rho_l_input'], 'density'),
                            rho_g_fps=to_fps(inputs['rho_g_input'], 'density'),
                            mu_g_fps=to_fps(inputs['mu_g_input'], 'viscosity'),
                            mist_extractor_type_str="Cyclonic", # Pass the type string
                            cyclone_type_params_fps=cyclone_type_params_with_user_inputs
                        )
                    else:
                        st.session_state.plot_data_after_mist_extractor = st.session_state.plot_data_after_gravity # No mist extractor selected, so no change
                    
                    # Update the overall mist extractor separation efficiency in results for reporting
                    if st.session_state.plot_data_after_mist_extractor:
                        st.session_state.calculation_results['mist_extractor_separation_efficiency'] = st.session_state.plot_data_after_mist_extractor['overall_separation_efficiency']
                    else:
                        st.session_state.calculation_results['mist_extractor_separation_efficiency'] = 0.0
                else:
                    st.session_state.plot_data_after_mist_extractor = None # No gravity data, so no mist extractor data either

            else:
                st.warning("Required liquid flow rate or density inputs are missing in session state. Please check 'Input Parameters' page.")
                st.session_state.plot_data_original = None
                st.session_state.plot_data_adjusted = None
                st.session_state.plot_data_after_gravity = None
                st.session_state.plot_data_after_mist_extractor = None

        except Exception as e:
            st.error(f"An error occurred during plot data calculation: {e}")
            st.session_state.plot_data_original = None
            st.session_state.plot_data_adjusted = None
            st.session_state.plot_data_after_gravity = None
            st.session_state.plot_data_after_mist_extractor = None
    else:
        st.warning("Please go to the 'Input Parameters' page and modify inputs to trigger calculations and generate the plot data.")


    if st.session_state.plot_data_original and st.session_state.plot_data_adjusted and st.session_state.plot_data_after_gravity and st.session_state.plot_data_after_mist_extractor:
        plot_data_original = st.session_state.plot_data_original
        plot_data_adjusted = st.session_state.plot_data_adjusted
        plot_data_after_gravity = st.session_state.plot_data_after_gravity
        plot_data_after_mist_extractor = st.session_state.plot_data_after_mist_extractor
        
        # Define unit labels for plotting
        micron_unit_label = "µm" # Always SI for this version
        mass_flow_unit = "kg/s"
        vol_flow_unit = "m³/s" # New unit for Streamlit display

        # --- Plot for Original Distribution ---
        st.subheader("3.1. Distribution Before Inlet Device")
        dp_values_microns_original = plot_data_original['dp_values_ft'] * FT_TO_MICRON
        fig_original, ax_original = plt.subplots(figsize=(10, 6))

        ax_original.plot(dp_values_microns_original, plot_data_original['cumulative_volume_undersize'], 'o-', label='Cumulative Volume Undersize', markersize=2, color='#1f77b4')
        ax_original.plot(dp_values_microns_original, plot_data_original['cumulative_volume_oversize'], 'o-', label='Cumulative Volume Oversize', markersize=2, color='#d62728')
        ax_original.set_xlabel(f'Droplet Size ({micron_unit_label})', fontsize=12)
        ax_original.set_ylabel('Cumulative Volume Fraction', color='black', fontsize=12)
        ax_original.tick_params(axis='y', labelcolor='black')
        ax_original.set_ylim(0, 1.05)
        ax_original.set_xlim(0, max(dp_values_microns_original) * 1.1 if dp_values_microns_original.size > 0 else 1000)

        ax2_original = ax_original.twinx()
        ax2_original.plot(dp_values_microns_original, plot_data_original['volume_fraction'], 'o-', label='Volume/Mass Fraction', markersize=2, color='#2ca02c')
        ax2_original.set_ylabel('Volume/Mass Fraction', color='black', fontsize=12)
        ax2_original.tick_params(axis='y', labelcolor='black')
        max_norm_fv_original = max(plot_data_original['volume_fraction']) if plot_data_original['volume_fraction'].size > 0 else 0.1
        ax2_original.set_ylim(0, max_norm_fv_original * 1.2)

        lines_original, labels_original = ax_original.get_legend_handles_labels()
        lines2_original, labels2_original = ax2_original.get_legend_handles_labels()
        ax2_original.legend(lines_original + lines2_original, labels_original + labels2_original, loc='upper left', fontsize=10)

        plt.title('Entrainment Droplet Size Distribution (Before Inlet Device)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig_original)
        plt.close(fig_original) # Close the plot to free memory

        # --- Plot for Adjusted Distribution ---
        st.subheader("3.2. Distribution After Inlet Device (Shift Factor Applied)")
        dp_values_microns_adjusted = plot_data_adjusted['dp_values_ft'] * FT_TO_MICRON
        fig_adjusted, ax_adjusted = plt.subplots(figsize=(10, 6))

        ax_adjusted.plot(dp_values_microns_adjusted, plot_data_adjusted['cumulative_volume_undersize'], 'o-', label='Cumulative Volume Undersize', markersize=2, color='#1f77b4')
        ax_adjusted.plot(dp_values_microns_adjusted, plot_data_adjusted['cumulative_volume_oversize'], 'o-', label='Cumulative Volume Oversize', markersize=2, color='#d62728')
        ax_adjusted.set_xlabel(f'Droplet Size ({micron_unit_label})', fontsize=12)
        ax_adjusted.set_ylabel('Cumulative Volume Fraction', color='black', fontsize=12)
        ax_adjusted.tick_params(axis='y', labelcolor='black')
        ax_adjusted.set_ylim(0, 1.05)
        ax_adjusted.set_xlim(0, max(dp_values_microns_adjusted) * 1.1 if dp_values_microns_adjusted.size > 0 else 1000)

        ax2_adjusted = ax_adjusted.twinx()
        ax2_adjusted.plot(dp_values_microns_adjusted, plot_data_adjusted['volume_fraction'], 'o-', label='Volume/Mass Fraction', markersize=2, color='#2ca02c')
        ax2_adjusted.set_ylabel('Volume/Mass Fraction', color='black', fontsize=12)
        ax2_adjusted.tick_params(axis='y', labelcolor='black')
        max_norm_fv_adjusted = max(plot_data_adjusted['volume_fraction']) if plot_data_adjusted['volume_fraction'].size > 0 else 0.1
        ax2_adjusted.set_ylim(0, max_norm_fv_adjusted * 1.2)

        lines_adjusted, labels_adjusted = ax_adjusted.get_legend_handles_labels()
        lines2_adjusted, labels2_adjusted = ax2_adjusted.get_legend_handles_labels()
        ax2_adjusted.legend(lines_adjusted + lines2_adjusted, labels_adjusted + labels2_adjusted, loc='upper left', fontsize=10)

        plt.title('Entrainment Droplet Size Distribution (After Inlet Device)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig_adjusted)
        plt.close(fig_adjusted) # Close the plot to free memory

        # --- Plot for After Gravity Settling ---
        st.subheader("3.3. Distribution After Gas Gravity Settling")
        dp_values_microns_after_gravity = plot_data_after_gravity['dp_values_ft'] * FT_TO_MICRON
        fig_after_gravity, ax_after_gravity = plt.subplots(figsize=(10, 6))

        ax_after_gravity.plot(dp_values_microns_after_gravity, plot_data_after_gravity['cumulative_volume_undersize'], 'o-', label='Cumulative Volume Undersize', markersize=2, color='#1f77b4')
        ax_after_gravity.plot(dp_values_microns_after_gravity, plot_data_after_gravity['cumulative_volume_oversize'], 'o-', label='Cumulative Volume Oversize', markersize=2, color='#d62728')
        ax_after_gravity.set_xlabel(f'Droplet Size ({micron_unit_label})', fontsize=12)
        ax_after_gravity.set_ylabel('Cumulative Volume Fraction', color='black', fontsize=12)
        ax_after_gravity.tick_params(axis='y', labelcolor='black')
        ax_after_gravity.set_ylim(0, 1.05)
        ax_after_gravity.set_xlim(0, max(dp_values_microns_after_gravity) * 1.1 if dp_values_microns_after_gravity.size > 0 else 1000)

        ax2_after_gravity = ax_after_gravity.twinx()
        ax2_after_gravity.plot(dp_values_microns_after_gravity, plot_data_after_gravity['volume_fraction'], 'o-', label='Volume/Mass Fraction', markersize=2, color='#2ca02c')
        ax2_after_gravity.set_ylabel('Volume/Mass Fraction', color='black', fontsize=12)
        ax2_after_gravity.tick_params(axis='y', labelcolor='black')
        max_norm_fv_after_gravity = max(plot_data_after_gravity['volume_fraction']) if plot_data_after_gravity['volume_fraction'].size > 0 else 0.1
        ax2_after_gravity.set_ylim(0, max_norm_fv_after_gravity * 1.2)

        lines_after_gravity, labels_after_gravity = ax_after_gravity.get_legend_handles_labels()
        lines2_after_gravity, labels2_after_gravity = ax2_after_gravity.get_legend_handles_labels()
        ax2_after_gravity.legend(lines_after_gravity + lines2_after_gravity, labels_after_gravity + labels2_after_gravity, loc='upper left', fontsize=10)

        plt.title('Entrainment Droplet Size Distribution (After Gravity Settling)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig_after_gravity)
        plt.close(fig_after_gravity) # Close the plot to free memory

        # --- Plot for After Mist Extractor ---
        st.subheader("3.4. Distribution After Mist Extractor")
        dp_values_microns_after_me = plot_data_after_mist_extractor['dp_values_ft'] * FT_TO_MICRON
        fig_after_me, ax_after_me = plt.subplots(figsize=(10, 6))

        ax_after_me.plot(dp_values_microns_after_me, plot_data_after_mist_extractor['cumulative_volume_undersize'], 'o-', label='Cumulative Volume Undersize', markersize=2, color='#1f77b4')
        ax_after_me.plot(dp_values_microns_after_me, plot_data_after_mist_extractor['cumulative_volume_oversize'], 'o-', label='Cumulative Volume Oversize', markersize=2, color='#d62728')
        ax_after_me.set_xlabel(f'Droplet Size ({micron_unit_label})', fontsize=12)
        ax_after_me.set_ylabel('Cumulative Volume Fraction', color='black', fontsize=12)
        ax_after_me.tick_params(axis='y', labelcolor='black')
        ax_after_me.set_ylim(0, 1.05)
        ax_after_me.set_xlim(0, max(dp_values_microns_after_me) * 1.1 if dp_values_microns_after_me.size > 0 else 1000)

        ax2_after_me = ax_after_me.twinx()
        ax2_after_me.plot(dp_values_microns_after_me, plot_data_after_mist_extractor['volume_fraction'], 'o-', label='Volume/Mass Fraction', markersize=2, color='#2ca02c')
        ax2_after_me.set_ylabel('Volume/Mass Fraction', color='black', fontsize=12)
        ax2_after_me.tick_params(axis='y', labelcolor='black')
        max_norm_fv_after_me = max(plot_data_after_mist_extractor['volume_fraction']) if plot_data_after_mist_extractor['volume_fraction'].size > 0 else 0.1
        ax2_after_me.set_ylim(0, max_norm_fv_after_me * 1.2)

        lines_after_me, labels_after_me = ax_after_me.get_legend_handles_labels()
        lines2_after_me, labels2_after_me = ax2_after_me.get_legend_handles_labels()
        ax2_after_me.legend(lines_after_me + lines2_after_me, labels_after_me + labels2_after_me, loc='upper left', fontsize=10)

        plt.title('Entrainment Droplet Size Distribution (After Mist Extractor)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig_after_me)
        plt.close(fig_after_me) # Close the plot to free memory


        # --- Volume Fraction Data Tables for Streamlit App ---
        st.subheader("4. Volume Fraction Data Tables (All Points)")

        # Original Data Table
        st.markdown("#### 4.1. Distribution Before Inlet Device")
        if plot_data_original['dp_values_ft'].size > 0:
            full_df_original = pd.DataFrame({
                "Droplet Size (µm)": plot_data_original['dp_values_ft'] * FT_TO_MICRON,
                "Volume Fraction": plot_data_original['volume_fraction'],
                "Cumulative Undersize": plot_data_original['cumulative_volume_undersize'],
                f"Entrained Mass Flow ({mass_flow_unit})": plot_data_original['entrained_mass_flow_rate_per_dp'],
                f"Entrained Volume Flow ({vol_flow_unit})": plot_data_original['entrained_volume_flow_rate_per_dp']
            })
            st.dataframe(full_df_original.style.format({
                "Droplet Size (µm)": "{:.2f}",
                "Volume Fraction": "{:.4f}",
                "Cumulative Undersize": "{:.4f}",
                f"Entrained Mass Flow ({mass_flow_unit})": "{:.6f}",
                f"Entrained Volume Flow ({vol_flow_unit})": "{:.9f}"
            }))
            st.markdown(f"**Sum of Entrained Mass Flow in Table:** {np.sum(plot_data_original['entrained_mass_flow_rate_per_dp']):.6f} {mass_flow_unit}")
            st.markdown(f"**Total Entrained Liquid Mass Flow Rate (Step 6):** {st.session_state.calculation_results['Q_entrained_total_mass_flow_rate_si']:.6f} {mass_flow_unit}")
            st.markdown(f"**Sum of Entrained Volume Flow in Table:** {np.sum(plot_data_original['entrained_volume_flow_rate_per_dp']):.9f} {vol_flow_unit}")
            st.markdown(f"**Total Entrained Liquid Volume Flow Rate (Step 6):** {st.session_state.calculation_results['Q_entrained_total_volume_flow_rate_si']:.9f} {vol_flow_unit}")
            st.info("Note: The sum of 'Entrained Flow' in the table will approximate the 'Total Entrained Liquid Flow Rate' calculated in Step 6, with the full sum matching if all data points were included.")
        else:
            st.info("No data available to display in the table for original distribution. Please check your input parameters.")

        # Adjusted Data Table
        st.markdown("#### 4.2. Distribution After Inlet Device (Shift Factor Applied)")
        if plot_data_adjusted['dp_values_ft'].size > 0:
            full_df_adjusted = pd.DataFrame({
                "Droplet Size (µm)": plot_data_adjusted['dp_values_ft'] * FT_TO_MICRON,
                "Volume Fraction": plot_data_adjusted['volume_fraction'],
                "Cumulative Undersize": plot_data_adjusted['cumulative_volume_undersize'],
                f"Entrained Mass Flow ({mass_flow_unit})": plot_data_adjusted['entrained_mass_flow_rate_per_dp'],
                f"Entrained Volume Flow ({vol_flow_unit})": plot_data_adjusted['entrained_volume_flow_rate_per_dp']
            })
            st.dataframe(full_df_adjusted.style.format({
                "Droplet Size (µm)": "{:.2f}",
                "Volume Fraction": "{:.4f}",
                "Cumulative Undersize": "{:.4f}",
                f"Entrained Mass Flow ({mass_flow_unit})": "{:.6f}",
                f"Entrained Volume Flow ({vol_flow_unit})": "{:.9f}"
            }))
            st.markdown(f"**Sum of Entrained Mass Flow in Table:** {np.sum(plot_data_adjusted['entrained_mass_flow_rate_per_dp']):.6f} {mass_flow_unit}")
            st.markdown(f"**Total Entrained Liquid Mass Flow Rate (from previous stage):** {plot_data_original['total_entrained_mass_flow_rate_si']:.6f} {mass_flow_unit}")
            st.markdown(f"**Sum of Entrained Volume Flow in Table:** {np.sum(plot_data_adjusted['entrained_volume_flow_rate_per_dp']):.9f} {vol_flow_unit}")
            st.markdown(f"**Total Entrained Liquid Volume Flow Rate (from previous stage):** {plot_data_original['total_entrained_volume_flow_rate_si']:.9f} {vol_flow_unit}")
            st.info("Note: The sum of 'Entrained Flow' in the table will approximate the 'Total Entrained Liquid Flow Rate' from the previous stage, with the full sum matching if all data points were included.")
        else:
            st.info("No data available to display in the table for adjusted distribution. Please check your input parameters.")

        # Data Table After Gravity Settling
        st.markdown("#### 4.3. Distribution After Gas Gravity Settling")
        if plot_data_after_gravity['dp_values_ft'].size > 0:
            full_df_after_gravity = pd.DataFrame({
                "Droplet Size (µm)": plot_data_after_gravity['dp_values_ft'] * FT_TO_MICRON,
                "Volume Fraction": plot_data_after_gravity['volume_fraction'],
                "Cumulative Undersize": plot_data_after_gravity['cumulative_volume_undersize'],
                f"Entrained Mass Flow ({mass_flow_unit})": plot_data_after_gravity['entrained_mass_flow_rate_per_dp'],
                f"Entrained Volume Flow ({vol_flow_unit})": plot_data_after_gravity['entrained_volume_flow_rate_per_dp']
            })
            st.dataframe(full_df_after_gravity.style.format({
                "Droplet Size (µm)": "{:.2f}",
                "Volume Fraction": "{:.4f}",
                "Cumulative Undersize": "{:.4f}",
                f"Entrained Mass Flow ({mass_flow_unit})": "{:.6f}",
                f"Entrained Volume Flow ({vol_flow_unit})": "{:.9f}"
            }))
            st.markdown(f"**Sum of Entrained Mass Flow in Table:** {np.sum(plot_data_after_gravity['entrained_mass_flow_rate_per_dp']):.6f} {mass_flow_unit}")
            st.markdown(f"**Total Entrained Liquid Mass Flow Rate (from previous stage):** {plot_data_adjusted['total_entrained_mass_flow_rate_si']:.6f} {mass_flow_unit}")
            st.markdown(f"**Sum of Entrained Volume Flow in Table:** {np.sum(plot_data_after_gravity['entrained_volume_flow_rate_per_dp']):.9f} {vol_flow_unit}")
            st.markdown(f"**Total Entrained Liquid Volume Flow Rate (from previous stage):** {plot_data_adjusted['total_entrained_volume_flow_rate_si']:.9f} {vol_flow_unit}")
            st.info("Note: The sum of 'Entrained Flow' in the table will approximate the 'Total Entrained Liquid Flow Rate' from the previous stage, with the full sum matching if all data points were included.")
        else:
            st.info("No data available to display in the table for gravity settling. Please check your input parameters.")

        # Data Table After Mist Extractor
        st.markdown("#### 4.4. Distribution After Mist Extractor")
        if plot_data_after_mist_extractor['dp_values_ft'].size > 0:
            full_df_after_me = pd.DataFrame({
                "Droplet Size (µm)": plot_data_after_mist_extractor['dp_values_ft'] * FT_TO_MICRON,
                "Volume Fraction": plot_data_after_mist_extractor['volume_fraction'],
                "Cumulative Undersize": plot_data_after_mist_extractor['cumulative_volume_undersize'],
                f"Entrained Mass Flow ({mass_flow_unit})": plot_data_after_mist_extractor['entrained_mass_flow_rate_per_dp'],
                f"Entrained Volume Flow ({vol_flow_unit})": plot_data_after_mist_extractor['entrained_volume_flow_rate_per_dp']
            })
            st.dataframe(full_df_after_me.style.format({
                "Droplet Size (µm)": "{:.2f}",
                "Volume Fraction": "{:.4f}",
                "Cumulative Undersize": "{:.4f}",
                f"Entrained Mass Flow ({mass_flow_unit})": "{:.6f}",
                f"Entrained Volume Flow ({vol_flow_unit})": "{:.9f}"
            }))
            st.markdown(f"**Sum of Entrained Mass Flow in Table:** {np.sum(plot_data_after_mist_extractor['entrained_mass_flow_rate_per_dp']):.6f} {mass_flow_unit}")
            st.markdown(f"**Total Entrained Liquid Mass Flow Rate (from previous stage):** {plot_data_after_gravity['total_entrained_mass_flow_rate_si']:.6f} {mass_flow_unit}")
            st.markdown(f"**Sum of Entrained Volume Flow in Table:** {np.sum(plot_data_after_mist_extractor['entrained_volume_flow_rate_per_dp']):.9f} {vol_flow_unit}")
            st.markdown(f"**Total Entrained Liquid Volume Flow Rate (from previous stage):** {plot_data_after_gravity['total_entrained_volume_flow_rate_si']:.9f} {vol_flow_unit}")
            st.info("Note: The sum of 'Entrained Flow' in the table will approximate the 'Total Entrained Liquid Flow Rate' from the previous stage, with the full sum matching if all data points were included.")
        else:
            st.info("No data available to display in the table for mist extractor. Please check your input parameters.")


        # Save plots to BytesIO objects for PDF embedding
        buf_original = io.BytesIO()
        fig_original = plt.figure(figsize=(10, 6)) # Recreate figure for saving
        ax_original = fig_original.add_subplot(111)
        ax_original.plot(dp_values_microns_original, plot_data_original['cumulative_volume_undersize'], 'o-', label='Cumulative Volume Undersize', markersize=2, color='#1f77b4')
        ax_original.plot(dp_values_microns_original, plot_data_original['cumulative_volume_oversize'], 'o-', label='Cumulative Volume Oversize', markersize=2, color='#d62728')
        ax_original.set_xlabel(f'Droplet Size ({micron_unit_label})', fontsize=12)
        ax_original.set_ylabel('Cumulative Volume Fraction', color='black', fontsize=12)
        ax_original.tick_params(axis='y', labelcolor='black')
        ax_original.set_ylim(0, 1.05)
        ax_original.set_xlim(0, max(dp_values_microns_original) * 1.1 if dp_values_microns_original.size > 0 else 1000)
        ax2_original = ax_original.twinx()
        ax2_original.plot(dp_values_microns_original, plot_data_original['volume_fraction'], 'o-', label='Volume/Mass Fraction', markersize=2, color='#2ca02c')
        ax2_original.set_ylabel('Volume/Mass Fraction', color='black', fontsize=12)
        ax2_original.tick_params(axis='y', labelcolor='black')
        max_norm_fv_original = max(plot_data_original['volume_fraction']) if plot_data_original['volume_fraction'].size > 0 else 0.1
        ax2_original.set_ylim(0, max_norm_fv_original * 1.2)
        lines_original, labels_original = ax_original.get_legend_handles_labels()
        lines2_original, labels2_original = ax2_original.get_legend_handles_labels()
        ax2_original.legend(lines_original + lines2_original, labels_original + labels2_original, loc='upper left', fontsize=10)
        plt.title('Entrainment Droplet Size Distribution (Before Inlet Device)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        fig_original.savefig(buf_original, format="png", dpi=300)
        buf_original.seek(0)
        plt.close(fig_original) # Close the plot to free memory


        buf_adjusted = io.BytesIO()
        fig_adjusted = plt.figure(figsize=(10, 6)) # Recreate figure for saving
        ax_adjusted = fig_adjusted.add_subplot(111)
        ax_adjusted.plot(dp_values_microns_adjusted, plot_data_adjusted['cumulative_volume_undersize'], 'o-', label='Cumulative Volume Undersize', markersize=2, color='#1f77b4')
        ax_adjusted.plot(dp_values_microns_adjusted, plot_data_adjusted['cumulative_volume_oversize'], 'o-', label='Cumulative Volume Oversize', markersize=2, color='#d62728')
        ax_adjusted.set_xlabel(f'Droplet Size ({micron_unit_label})', fontsize=12)
        ax_adjusted.set_ylabel('Cumulative Volume Fraction', color='black', fontsize=12)
        ax_adjusted.tick_params(axis='y', labelcolor='black')
        ax_adjusted.set_ylim(0, 1.05)
        ax_adjusted.set_xlim(0, max(dp_values_microns_adjusted) * 1.1 if dp_values_microns_adjusted.size > 0 else 1000)
        ax2_adjusted = ax_adjusted.twinx()
        ax2_adjusted.plot(dp_values_microns_adjusted, plot_data_adjusted['volume_fraction'], 'o-', label='Volume/Mass Fraction', markersize=2, color='#2ca02c')
        ax2_adjusted.set_ylabel('Volume/Mass Fraction', color='black', fontsize=12)
        ax2_adjusted.tick_params(axis='y', labelcolor='black')
        max_norm_fv_adjusted = max(plot_data_adjusted['volume_fraction']) if plot_data_adjusted['volume_fraction'].size > 0 else 0.1
        ax2_adjusted.set_ylim(0, max_norm_fv_adjusted * 1.2)
        lines_adjusted, labels_adjusted = ax_adjusted.get_legend_handles_labels()
        lines2_adjusted, labels2_adjusted = ax2_adjusted.get_legend_handles_labels()
        ax2_adjusted.legend(lines_adjusted + lines2_adjusted, labels_adjusted + labels2_adjusted, loc='upper left', fontsize=10)
        plt.title('Entrainment Droplet Size Distribution (After Inlet Device)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        fig_adjusted.savefig(buf_adjusted, format="png", dpi=300)
        buf_adjusted.seek(0)
        plt.close(fig_adjusted) # Close the plot to free memory


        buf_after_gravity = io.BytesIO()
        fig_after_gravity = plt.figure(figsize=(10, 6)) # Recreate figure for saving
        ax_after_gravity = fig_after_gravity.add_subplot(111)
        ax_after_gravity.plot(dp_values_microns_after_gravity, plot_data_after_gravity['cumulative_volume_undersize'], 'o-', label='Cumulative Volume Undersize', markersize=2, color='#1f77b4')
        ax_after_gravity.plot(dp_values_microns_after_gravity, plot_data_after_gravity['cumulative_volume_oversize'], 'o-', label='Cumulative Volume Oversize', markersize=2, color='#d62728')
        ax_after_gravity.set_xlabel(f'Droplet Size ({micron_unit_label})', fontsize=12)
        ax_after_gravity.set_ylabel('Cumulative Volume Fraction', color='black', fontsize=12)
        ax_after_gravity.tick_params(axis='y', labelcolor='black')
        ax_after_gravity.set_ylim(0, 1.05)
        ax_after_gravity.set_xlim(0, max(dp_values_microns_after_gravity) * 1.1 if dp_values_microns_after_gravity.size > 0 else 1000)
        ax2_after_gravity = ax_after_gravity.twinx()
        ax2_after_gravity.plot(dp_values_microns_after_gravity, plot_data_after_gravity['volume_fraction'], 'o-', label='Volume/Mass Fraction', markersize=2, color='#2ca02c')
        ax2_after_gravity.set_ylabel('Volume/Mass Fraction', color='black', fontsize=12)
        ax2_after_gravity.tick_params(axis='y', labelcolor='black')
        max_norm_fv_after_gravity = max(plot_data_after_gravity['volume_fraction']) if plot_data_after_gravity['volume_fraction'].size > 0 else 0.1
        ax2_after_gravity.set_ylim(0, max_norm_fv_after_gravity * 1.2)
        lines_after_gravity, labels_after_gravity = ax_after_gravity.get_legend_handles_labels()
        lines2_after_gravity, labels2_after_gravity = ax2_after_gravity.get_legend_handles_labels()
        ax2_after_gravity.legend(lines_after_gravity + lines2_after_gravity, labels_after_gravity + labels2_after_gravity, loc='upper left', fontsize=10)
        plt.title('Entrainment Droplet Size Distribution (After Gravity Settling)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        fig_after_gravity.savefig(buf_after_gravity, format="png", dpi=300)
        buf_after_gravity.seek(0)
        plt.close(fig_after_gravity) # Close the plot to free memory

        buf_after_me = io.BytesIO()
        fig_after_me = plt.figure(figsize=(10, 6)) # Recreate figure for saving
        ax_after_me = fig_after_me.add_subplot(111)
        ax_after_me.plot(dp_values_microns_after_me, plot_data_after_mist_extractor['cumulative_volume_undersize'], 'o-', label='Cumulative Volume Undersize', markersize=2, color='#1f77b4')
        ax_after_me.plot(dp_values_microns_after_me, plot_data_after_mist_extractor['cumulative_volume_oversize'], 'o-', label='Cumulative Volume Oversize', markersize=2, color='#d62728')
        ax_after_me.set_xlabel(f'Droplet Size ({micron_unit_label})', fontsize=12)
        ax_after_me.set_ylabel('Cumulative Volume Fraction', color='black', fontsize=12)
        ax_after_me.tick_params(axis='y', labelcolor='black')
        ax_after_me.set_ylim(0, 1.05)
        ax_after_me.set_xlim(0, max(dp_values_microns_after_me) * 1.1 if dp_values_microns_after_me.size > 0 else 1000)
        ax2_after_me = ax_after_me.twinx()
        ax2_after_me.plot(dp_values_microns_after_me, plot_data_after_mist_extractor['volume_fraction'], 'o-', label='Volume/Mass Fraction', markersize=2, color='#2ca02c')
        ax2_after_me.set_ylabel('Volume/Mass Fraction', color='black', fontsize=12)
        ax2_after_me.tick_params(axis='y', labelcolor='black')
        max_norm_fv_after_me = max(plot_data_after_mist_extractor['volume_fraction']) if plot_data_after_mist_extractor['volume_fraction'].size > 0 else 0.1
        ax2_after_me.set_ylim(0, max_norm_fv_after_me * 1.2)
        lines_after_me, labels_after_me = ax_after_me.get_legend_handles_labels()
        lines2_after_me, labels2_after_me = ax2_after_me.get_legend_handles_labels()
        ax2_after_me.legend(lines_after_me + lines2_after_me, labels_after_me + labels2_after_me, loc='upper left', fontsize=10)
        plt.title('Entrainment Droplet Size Distribution (After Mist Extractor)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        fig_after_me.savefig(buf_after_me, format="png", dpi=300)
        buf_after_me.seek(0)
        plt.close(fig_after_me) # Close the plot to free memory


        st.download_button(
            label="Download Report as PDF",
            data=generate_pdf_report(
                st.session_state.inputs,
                st.session_state.calculation_results,
                buf_original,
                buf_adjusted,
                plot_data_original,
                plot_data_adjusted,
                plot_data_after_gravity,
                plot_data_after_mist_extractor
            ),
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
* **Volume/Mass Fraction Distribution:** The 'Volume Fraction' and 'Mass Fraction' values displayed in the table and plot now represent the **normalized volume frequency distribution** as stated in the article. This means the sum of these fractions over the entire distribution range is 1. Consequently, the sum of 'Entrained Flow (kg/s)' for sampled points in the table will approximate the 'Total Entrained Liquid Flow Rate' calculated in Step 6, with the full sum matching if all data points were included.
""")
