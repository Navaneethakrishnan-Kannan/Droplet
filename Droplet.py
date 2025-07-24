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
# a = 4.0 implies d_max = 5.0 * d_v50 (since a = (d_max - d_v50) / d_v50)
A_DISTRIBUTION = 4.0
# The article states delta = 0.72 as a typical value.
# The formula provided for delta appears to be a typo (V_50/V_50).
DELTA_DISTRIBUTION = 0.72

# Figure 9 Approximation for Droplet Size Distribution Shift Factor
# This function approximates the "Liquid Separation Efficiency/Droplet Size Distribution Shift Factor"
# curves from Figure 9 of the article. It's a simplified interpretation based on visual data.
# A more accurate implementation would require digitizing the curves precisely.
# The shift factor directly multiplies the calculated d_v50. A factor less than 1 indicates
# smaller droplets due to the inlet device's shattering/shearing effect.
def get_shift_factor(inlet_device, rho_v_squared):
    """
    Approximates the droplet size distribution shift factor based on the inlet device
    and inlet momentum (rho_g * V_g^2).
    """
    if inlet_device == "No inlet device":
        return 1.0 # No shift if no inlet device
    elif inlet_device == "Diverter plate":
        # Approximated piecewise linear curve from Fig. 9
        if rho_v_squared <= 1000:
            return 1.0 - (0.1 * rho_v_squared / 1000)
        elif rho_v_squared <= 2000:
            return 0.9 - (0.3 * (rho_v_squared - 1000) / 1000)
        elif rho_v_squared <= 3000:
            return 0.6 - (0.4 * (rho_v_squared - 2000) / 1000)
        else: # Beyond the typical effective range for this device
            return 0.2
    elif inlet_device == "Half-pipe":
        # Approximated piecewise linear curve from Fig. 9
        if rho_v_squared <= 1000:
            return 1.0 - (0.1 * rho_v_squared / 1000)
        elif rho_v_squared <= 2000:
            return 0.9 - (0.2 * (rho_v_squared - 1000) / 1000)
        elif rho_v_squared <= 3000:
            return 0.7 - (0.2 * (rho_v_squared - 2000) / 1000)
        elif rho_v_squared <= 4000:
            return 0.5 - (0.2 * (rho_v_squared - 3000) / 1000)
        else:
            return 0.2 # Placeholder for very high values
    elif inlet_device == "Vane-type":
        # Approximated piecewise linear curve from Fig. 9
        if rho_v_squared <= 5000:
            return 1.0 - (0.05 * rho_v_squared / 5000)
        elif rho_v_squared <= 7000:
            return 0.95 - (0.15 * (rho_v_squared - 5000) / 2000)
        elif rho_v_squared <= 10000:
            return 0.8 - (0.3 * (rho_v_squared - 7000) / 3000)
        else:
            return 0.5 # Placeholder for very high values
    elif inlet_device == "Cyclonic":
        # Approximated piecewise linear curve from Fig. 9
        if rho_v_squared <= 10000:
            return 1.0 - (0.1 * rho_v_squared / 10000)
        elif rho_v_squared <= 14000:
            return 0.9 - (0.7 * (rho_v_squared - 10000) / 4000)
        else:
            return 0.2 # Placeholder for very high values
    return 1.0 # Default if an unknown inlet device is somehow selected

# --- Streamlit App Layout ---

st.set_page_config(layout="centered", page_title="Oil & Gas Separation App")

st.title("🛢️ Oil and Gas Separation: Particle Size Distribution")
st.markdown("""
This application helps quantify the entrained liquid droplet size distribution in gas-liquid separators,
based on the principles and correlations discussed in the article "Quantifying Separation Performance" by Mark Bothamley.
""")

st.header("1. Input Parameters (FPS Units)")

# Section for Feed Pipe Conditions
st.subheader("Feed Pipe Conditions")
col1, col2 = st.columns(2)
with col1:
    D = st.number_input("Pipe Inside Diameter (ft)", min_value=0.01, value=1.0, format="%.2f",
                        help="Diameter of the feed pipe to the separator.")
    rho_l = st.number_input("Liquid Density (lb/ft³)", min_value=0.01, value=40.0, format="%.2f",
                            help="Density of the liquid phase.")
    mu_l = st.number_input("Liquid Viscosity (lb/ft-sec)", min_value=1e-7, value=0.0005, format="%.7f",
                            help="Viscosity of the liquid phase. Example: Water at 60°F is ~0.00075 lb/ft-sec.")
with col2:
    V_g = st.number_input("Gas Velocity (ft/sec)", min_value=0.01, value=50.0, format="%.2f",
                          help="Superficial gas velocity in the feed pipe.")
    rho_g = st.number_input("Gas Density (lb/ft³)", min_value=1e-4, value=0.1, format="%.4f",
                            help="Density of the gas phase.")
    mu_g = st.number_input("Gas Viscosity (lb/ft-sec)", min_value=1e-8, value=0.00001, format="%.8f",
                            help="Viscosity of the gas phase. Example: Methane at 60°F is ~0.000007 lb/ft-sec.")

# New columns for Liquid Surface Tension and Separator Inlet Device
st.markdown("---") # Separator for visual clarity
col_st, col_id = st.columns(2)

with col_st:
    st.subheader("Liquid Surface Tension")
    surface_tension_option = st.selectbox(
        "Select Fluid System for Surface Tension",
        options=list(SURFACE_TENSION_TABLE.keys()) + ["Custom"],
        index=0,
        help="Choose a typical value or enter a custom one."
    )

    sigma_dyne_cm = 0.0
    if surface_tension_option == "Custom":
        sigma_dyne_cm = st.number_input("Custom Liquid Surface Tension (dyne/cm)", min_value=0.01, value=30.0, format="%.2f")
    else:
        sigma_dyne_cm = SURFACE_TENSION_TABLE[surface_tension_option]

    sigma = sigma_dyne_cm * DYNE_CM_TO_POUNDAL_FT
    st.info(f"**Calculated Liquid Surface Tension:** {sigma:.4f} poundal/ft")

with col_id:
    st.subheader("Separator Inlet Device")
    inlet_device = st.selectbox(
        "Choose Inlet Device Type",
        options=["No inlet device", "Diverter plate", "Half-pipe", "Vane-type", "Cyclonic"],
        help="The inlet device influences the droplet size distribution downstream."
    )

st.markdown("---") # Separator for visual clarity

# --- Calculation Trigger ---
if st.button("Calculate Particle Size Distribution", help="Click to perform calculations and view results."):
    st.header("2. Step-by-Step Calculation Results")

    try:
        # Step 1: Calculate Superficial Gas Reynolds Number (Re_g)
        st.markdown("#### Step 1: Calculate Superficial Gas Reynolds Number ($Re_g$)")
        if mu_g == 0:
            st.error("Gas viscosity (μg) cannot be zero for Reynolds number calculation.")
            st.stop()
        Re_g = (D * V_g * rho_g) / mu_g
        st.write(f"Equation: $Re_g = \\frac{{D \\cdot V_g \\cdot \\rho_g}}{{\\mu_g}}$")
        st.write(f"Calculation: $Re_g = \\frac{{{D:.2f} \\text{{ ft}} \\cdot {V_g:.2f} \\text{{ ft/sec}} \\cdot {rho_g:.4f} \\text{{ lb/ft}}^3}}{{{mu_g:.8f} \\text{{ lb/ft-sec}}}} = {Re_g:.2f}$")
        st.success(f"**Result:** Superficial Gas Reynolds Number ($Re_g$) = **{Re_g:.2f}** (dimensionless)")

        st.markdown("---")

        # Step 2: Calculate Volume Median Diameter ($d_{v50}$) without inlet device effect
        st.markdown("#### Step 2: Calculate Initial Volume Median Diameter ($d_{v50}$) (Kataoka et al., 1983)")
        if V_g == 0 or rho_g == 0 or rho_l == 0 or mu_l == 0:
            st.error("Gas velocity, gas density, liquid density, and liquid viscosity must be non-zero for $d_{v50}$ calculation.")
            st.stop()
        
        # Kataoka et al. (1983) equation:
        # d_v50 = 0.01 * (sigma / (rho_g * V_g^2)) * Re_g^2 * (rho_g / rho_l)^(-1/3) * (mu_g / mu_l)^(2/3)
        dv50_original = 0.01 * (sigma / (rho_g * V_g**2)) * (Re_g**2) * ((rho_g / rho_l)**(-1/3)) * ((mu_g / mu_l)**(2/3))
        
        st.write(f"Equation: $d_{{v50}} = 0.01 \\left(\\frac{{\\sigma}}{{\\rho_g V_g^2}}\\right) Re_g^2 \\left(\\frac{{\\rho_g}}{{\\rho_l}}\\right)^{{-1/3}} \\left(\\frac{{\\mu_g}}{{\\mu_l}}\\right)^{{2/3}}$")
        st.write(f"Calculation: $d_{{v50}} = 0.01 \\left(\\frac{{{sigma:.4f}}}{{{rho_g:.4f} \\cdot ({V_g:.2f})^2}}\\right) ({Re_g:.2f})^2 \\left(\\frac{{{rho_g:.4f}}}{{{rho_l:.2f}}}\\right)^{{-0.333}} \\left(\\frac{{{mu_g:.8f}}}{{{mu_l:.7f}}}\\right)^{{0.667}}$")
        st.success(f"**Result:** Initial Volume Median Diameter ($d_{{v50}}$) = **{dv50_original * 304800:.2f} microns** ({dv50_original:.6f} ft)")

        st.markdown("---")

        # Step 3: Determine Inlet Momentum (rho_g V_g^2)
        st.markdown("#### Step 3: Calculate Inlet Momentum ($\\rho_g V_g^2$)")
        rho_v_squared = rho_g * V_g**2
        st.write(f"Equation: $\\rho_g V_g^2 = \\rho_g \\cdot V_g^2$")
        st.write(f"Calculation: $\\rho_g V_g^2 = {rho_g:.4f} \\text{{ lb/ft}}^3 \\cdot ({V_g:.2f} \\text{{ ft/sec}})^2 = {rho_v_squared:.2f} \\text{{ lb/ft-sec}}^2$")
        st.success(f"**Result:** Inlet Momentum ($\\rho_g V_g^2$) = **{rho_v_squared:.2f} lb/ft-sec²**")

        st.markdown("---")

        # Step 4: Apply Inlet Device "Droplet Size Distribution Shift Factor"
        st.markdown("#### Step 4: Apply Inlet Device Effect (Droplet Size Distribution Shift Factor)")
        st.write(f"Selected Inlet Device: **{inlet_device}**")
        shift_factor = get_shift_factor(inlet_device, rho_v_squared)
        dv50_adjusted = dv50_original * shift_factor
        st.write(f"Based on Figure 9 from the article, for an inlet momentum of {rho_v_squared:.2f} lb/ft-sec² and a '{inlet_device}' device, the estimated shift factor is **{shift_factor:.3f}**.")
        st.write(f"Equation: $d_{{v50, adjusted}} = d_{{v50, original}} \\cdot \\text{{Shift Factor}}$")
        st.write(f"Calculation: $d_{{v50, adjusted}} = {dv50_original:.6f} \\text{{ ft}} \\cdot {shift_factor:.3f} = {dv50_adjusted:.6f} \\text{{ ft}}$")
        st.success(f"**Result:** Adjusted Volume Median Diameter ($d_{{v50}}$) = **{dv50_adjusted * 304800:.2f} microns** ({dv50_adjusted:.6f} ft)")

        st.markdown("---")

        # Step 5: Calculate parameters for Upper-Limit Log Normal Distribution
        st.markdown("#### Step 5: Calculate Parameters for Upper-Limit Log Normal Distribution")
        d_max = A_DISTRIBUTION * dv50_adjusted # d_max = 5 * d_v50 based on a = 4.0
        st.write(f"Using typical values from the article: $a = {A_DISTRIBUTION}$ and $\\delta = {DELTA_DISTRIBUTION}$.")
        st.write(f"Equation: $d_{{max}} = a \\cdot d_{{v50, adjusted}}$")
        st.write(f"Calculation: $d_{{max}} = {A_DISTRIBUTION} \\cdot {dv50_adjusted:.6f} \\text{{ ft}} = {d_max:.6f} \\text{{ ft}}$")
        st.success(f"**Result:** Maximum Droplet Size ($d_{{max}}$) = **{d_max * 304800:.2f} microns** ({d_max:.6f} ft)")

        st.markdown("---")

        # Step 6: Calculate Volume Fraction and Cumulative Volume Fraction for a range of droplet sizes
        st.markdown("#### Step 6: Generate Droplet Size Distribution Data")
        # Define droplet sizes in ft for calculation, then convert to microns for plotting
        # Start slightly above zero to avoid log(0) and ensure dp < d_max for the formula
        dp_min_calc = dv50_adjusted * 0.01 # Start at 1% of adjusted dv50 for better curve definition
        dp_max_calc = d_max * 0.999 # End just before d_max to avoid division by zero in ln argument

        # Generate a range of droplet sizes for plotting
        dp_values_ft = np.linspace(dp_min_calc, dp_max_calc, 500)

        volume_fraction = []
        cumulative_volume_undersize = []

        for dp in dp_values_ft:
            # z = ln(a * dp / (d_max - dp))
            z_val = np.log((A_DISTRIBUTION * dp) / (d_max - dp))

            # f_v(dp) = (delta * d_max) / sqrt(pi * dp * (d_max - dp)) * exp(-delta^2 * z^2)
            fv_dp = (DELTA_DISTRIBUTION * d_max) / np.sqrt(np.pi * dp * (d_max - dp)) * np.exp(-DELTA_DISTRIBUTION**2 * z_val**2)
            volume_fraction.append(fv_dp)

            # V_under = 1 - 0.5 * (1 - erf(delta * z))
            v_under = 1 - 0.5 * (1 - erf(DELTA_DISTRIBUTION * z_val))
            cumulative_volume_undersize.append(v_under)
        
        # Convert dp_values to microns for plotting
        dp_values_microns = dp_values_ft * 304800

        # Create cumulative volume oversize (1 - cumulative_volume_undersize)
        cumulative_volume_oversize = [1 - v for v in cumulative_volume_undersize]
        st.success("Droplet size distribution data generated successfully.")

        st.markdown("---")

        # --- Plotting Results (Similar to Fig. 7) ---
        st.header("3. Particle Size Distribution Plot")

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot Cumulative Volume Undersize and Oversize on primary Y-axis
        ax1.plot(dp_values_microns, cumulative_volume_undersize, 'o-', label='Cumulative Volume Undersize', markersize=2, color='#1f77b4') # Blue
        ax1.plot(dp_values_microns, cumulative_volume_oversize, 'o-', label='Cumulative Volume Oversize', markersize=2, color='#d62728') # Red
        ax1.set_xlabel('Droplet Size (microns)', fontsize=12)
        ax1.set_ylabel('Cumulative Volume Fraction', color='black', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_ylim(0, 1.4) # To match the article's figure scale
        ax1.set_xlim(0, max(dp_values_microns) * 1.1 if dp_values_microns.size > 0 else 1000) # Extend x-axis slightly

        # Create a second Y-axis for Volume Fraction
        ax2 = ax1.twinx()
        ax2.plot(dp_values_microns, volume_fraction, 'o-', label='Volume Fraction', markersize=2, color='#2ca02c') # Green
        ax2.set_ylabel('Volume Fraction', color='black', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Dynamic adjustment for the second Y-axis limit
        max_fv = max(volume_fraction) if volume_fraction else 0.1
        ax2.set_ylim(0, max_fv * 1.2)

        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=10)

        plt.title('Entrainment Droplet Size Distribution', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred during calculation: {e}")
        st.warning("Please review your input values. Ensure they are realistic and non-zero where required to prevent mathematical errors.")

st.markdown("""
---
#### Important Notes:
* **Figure 9 Approximation:** The "Droplet Size Distribution Shift Factor" is based on a simplified interpretation of Figure 9 from the article. For highly precise engineering applications, the curves in Figure 9 would need to be digitized and accurately modeled.
* **Log Normal Distribution Parameters:** The article states typical values for $a=4.0$ and $\delta=0.72$. The formula for $\delta$ shown in the article ($\\delta=\\frac{0.394}{log(\\frac{V_{so}}{V_{so}})}$) appears to be a typographical error, so the constant value $\\delta=0.72$ is used as indicated in the text.
* **Units:** All inputs and calculations are performed using Foot-Pound-Second (FPS) units as specified in the article's tables.
""")
