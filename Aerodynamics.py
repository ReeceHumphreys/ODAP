import numpy as np
import pandas as pd

excel_file = "AtmosphericModelValues.xlsx"
earth_radius = 6378.0 #km
tabulated_values = pd.read_excel(excel_file, engine="openpyxl")

def atmosphere_density(altitude):
    bins = tabulated_values['Altitude Lower Bound (km)'].values
    base_altitude = tabulated_values['Base Altitude (km)'].values
    nominal_density = tabulated_values['Nominal Density (kg/m^3)'].values
    scale_height = tabulated_values['Scale Height (km)'].values
    i = np.digitize(altitude, bins) - 1
    return (nominal_density[i]*np.exp(-(altitude-base_altitude[i])/(scale_height[i])))

def scale_height(altitude):
    bins = tabulated_values['Altitude Lower Bound (km)'].values
    scale_height = tabulated_values['Scale Height (km)'].values
    i = np.digitize(altitude, bins) - 1
    return scale_height[i]

