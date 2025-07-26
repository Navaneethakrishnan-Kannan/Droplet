import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF

# --- Unit Conversion Factors (for internal FPS calculation) ---
# These constants are needed by generate_pdf_report for display conversions
M_TO_FT = 3.28084 # 1 meter = 3.28084 feet
FT_TO_MICRON = 1 / (1e-6 * M_TO_FT) # 1 ft = 1 / (1e-6 * M_TO_FT) microns
NM_TO_POUNDAL_FT = 2.2 # 1 N/m = 1000 dyne/cm; 1 dyne/cm = 0.0022 poundal/ft => 1 N/m = 2.2 poundal/ft
KG_M3_TO_LB_FT3 = 0.0624279 # 1 kg/m^3 = 0.0624279 lb/ft^3
MPS_TO_FTPS = 3.28084 # 1 m/s = 3.28084 ft/s
PAS_TO_LB_FT_S = 0.67197 # 1 Pa.s (kg/m.s) = 0.67197 lb/ft.s
IN_TO_FT = 1/12 # 1 inch = 1/12 feet

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

# Typical values for Upper-Limit Log Normal Distribution (from the article)
A_DISTRIBUTION = 4.0
DELTA_DISTRIBUTION = 0.72

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


def generate_pdf_report(inputs, results, plot_image_buffer_original, plot_image_buffer_adjusted, plot_data_original, plot_data_adjusted, plot_data_after_gravity, plot_data_after_mist_extractor, report_date):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title Page
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 20, 'Oil and Gas Separation: Particle Size Distribution Analysis', 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Date: {report_date}', 0, 1, 'C')
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

    # Step 8: Gas Gravity Separation Section Efficiency
    pdf.set_font('Arial', 'B', 10)
    pdf.chapter_body("Step 8: Gas Gravity Separation Section Efficiency")
    pdf.set_font('Arial', '', 10)
    if inputs['separator_type'] == "Horizontal":
        pdf.chapter_body(f"Separator Type: Horizontal")
        pdf.chapter_body(f"Gas Space Height (hg): {inputs['h_g_input']:.3f} {len_unit_pdf}")
        pdf.chapter_body(f"Effective Separation Length (Le): {inputs['L_e_input']:.3f} {len_unit_pdf}")
    else: # Vertical
        pdf.chapter_body(f"Separator Type: Vertical")
        pdf.chapter_body(f"Separator Diameter: {inputs['D_separator_input']:.3f} {len_unit_pdf}")
    pdf.chapter_body(f"Overall Separation Efficiency of Gravity Section: {results['gravity_separation_efficiency']:.2%}")
    pdf.chapter_body(f"Total Entrained Liquid Mass Flow Rate After Gravity Settling: {plot_data_after_gravity['total_entrained_mass_flow_rate_si']:.4f} {mass_flow_unit_pdf}")
    pdf.chapter_body(f"Total Entrained Liquid Volume Flow Rate After Gravity Settling: {plot_data_after_gravity['total_entrained_volume_flow_rate_si']:.6f} {vol_flow_unit_pdf}")
    pdf.ln(5)

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