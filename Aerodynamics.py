import numpy as np
import pandas as pd

excel_file = "AtmosphericModelValues.xlsx"
tabulated_values = pd.read_excel(excel_file, engine="openpyxl")

def atmosphere_density(altitude):
    z = altitude / 1e3 #[km] (Need to work in km due to given data format)
    bins = tabulated_values['Altitude Lower Bound (km)'].values
    base_altitude = tabulated_values['Base Altitude (km)'].values
    nominal_density = tabulated_values['Nominal Density (kg/m^3)'].values
    scale_height = tabulated_values['Scale Height (km)'].values
    i = np.digitize(z, bins) - 1
    result = (nominal_density[i]*np.exp(-(z-base_altitude[i])/(scale_height[i]))) #[kg•km^-3]
    return result #[kg•m^-3]

def scale_height(altitude):
    z = altitude / 1e3 #[km] (Need to work in km due to given data format)
    bins = tabulated_values['Altitude Lower Bound (km)'].values
    scale_height = tabulated_values['Scale Height (km)'].values
    i = np.digitize(z, bins) - 1
    result = scale_height[i] #[km]
    return result * 1e3 #[m]
