from .SimulationConfiguration import SimulationType
from .SimulationConfiguration import SatType
from .utils.utils import power_law
from .utils.AMUtils import mean_1, mean_2, mean_soc, sigma_1, sigma_2, sigma_soc, alpha
import numpy as np


class FragmentationEvent():

    _input_mass = 0
    _output = np.array([])

    # TODO: This is just for explosions, will need to refactor when adding collisions
    _lc_power_law_exponent = -2.6
    _deltaVelocityFactorOffset = [0.2, 1.85]

    # Sats is an array containing the satellites involved in the fragmentation event
    # Will contain one for explosions and two for collisions
    def __init__(self, config, sats):

        self.sats = sats

        # Explosion or Collision
        self._simulation_type = config.simulationType

        # Setting the mass before the fragmentation event
        self._input_mass = np.sum([sat.mass for sat in sats])

        # Categorization of satellite: RB, SC, or SOC
        self._sat_type = config.sat_type

        # Setting characteristic lengths
        self._min_characteristic_length = config.minimalCharacteristicLength
        self._max_characteristic_length = sats[0].characteristic_length

    def run(self):
        # Compute the number of fragments generate in the fragmentation event
        count = self._fragment_count(self._min_characteristic_length)
        # Location the explosion occured
        r = self.sats[0].position

        # Assigning debris type and location
        self._output = np.empty((count, 7, 3))
        self._output[:, 0] = SatType.deb.index
        self._output[:, 1] = r

        print(count)

        # Characteristic Length and AM for each debris
        for i in range(count):
            # Computing L_c for each debris following powerlaw
            self._output[i, 2] = self._characteristic_length_distribution()
            # Computing A/M ratio for debris
            self._output[i, 3] = self._AM_Ratio(self._output[i, 2, 0])
            # Computing Area for each debris using L_c
            self._output[i, 4] = self._compute_Area(self._output[i, 2, 0])
            # Compute Mass using area and AM ratio
            self._output[i, 5] = self._compute_mass(
                self._output[i, 4, 0], self._output[i, 3, 0])

        # Mass conservation
        self._conserve_mass()
        count = self._output.shape[0]

        # Determine debris velocity
        self._output[:, 6] = self.sats[0].velocity
        for i in range(count):
            chi = np.log10(self._output[i, 3, 0])
            mean = self._deltaVelocityFactorOffset[0] * \
                chi + self._deltaVelocityFactorOffset[1]
            sigma = 0.4
            n = np.random.normal(mean, sigma)
            velocity_scalar = pow(10.0, n)

            # Transform the scalar velocity into a vector
            ejection_velocity = self._velocity_vector(velocity_scalar)
            velocity = self._output[i, 6, :] + ejection_velocity
            self._output[i, 6] = velocity

        return self._output

    def _velocity_vector(self, velocity):
        n1 = np.random.uniform(0.0, 1)
        n2 = np.random.uniform(0.0, 1)
        u = n1 * 2.0 - 1.0
        theta = n2 * 2.0 * np.pi
        v = np.sqrt(1.0 - u * u)
        return np.array([v * np.cos(theta) * velocity, v * np.sin(theta) * velocity, u * velocity])

    def _conserve_mass(self):

        # Enforce Mass Conservation if the output mass is greater than the input mass
        output_mass = np.sum(self._output[:, 5, 0])
        old_length = self._output.shape[0]
        new_length = old_length

        while output_mass > self._input_mass:
            self._output = np.delete(self._output, -1, 0)
            output_mass = np.sum(self._output[:, 5, 0])
            new_length = self._output.shape[0]

        if old_length != new_length:
            print("TODO: Removed debris to bring output mass close to input")
        else:
            while self._input_mass > output_mass:
                new_row = np.empty((7, 3))
                new_row[0] = SatType.deb.index
                new_row[1] = self.sats[0].position
                # Computing L_c for each debris following powerlaw
                new_row[2] = self._characteristic_length_distribution()
                # Computing A/M ratio for debris
                new_row[3] = self._AM_Ratio(new_row[2, 0])
                # Computing Area for each debris using L_c
                new_row[4] = self._compute_Area(new_row[2, 0])
                # Compute Mass using area and AM ratio
                new_row[5] = self._compute_mass(new_row[4, 0], new_row[3, 0])
                self._output = np.insert(self._output, -1, new_row, 0)
                output_mass = np.sum(self._output[:, 5, 0])
                new_length = self._output.shape

            # Remove the element that causes output mass to be too large now
            self._output = np.delete(self._output, -1, 0)

    def _compute_mass(self, area, AM_ratio):
        return area / AM_ratio

    def _compute_Area(self, characteristic_length):
        l_c_bound = 0.00167
        if characteristic_length < l_c_bound:
            factor = 0.540424
            return factor * characteristic_length * characteristic_length
        else:
            exponent = 2.0047077
            factor = 0.556945
            return factor * pow(characteristic_length, exponent)

    def _AM_Ratio(self, characteristic_length):
        log_l_c = np.log10(characteristic_length)
        if characteristic_length > 0.11:
            # Case bigger than 11 cm
            n1 = np.random.normal(
                mean_1(self._sat_type, log_l_c), sigma_1(self._sat_type, log_l_c))
            n2 = np.random.normal(
                mean_2(self._sat_type, log_l_c), sigma_2(self._sat_type, log_l_c))

            return pow(10.0, alpha(self._sat_type, log_l_c) * n1 +
                       (1 - alpha(self._sat_type, log_l_c)) * n2)
        elif (characteristic_length < 0.08):
            # Case smaller than 8 cm
            n = np.random.normal(mean_soc(log_l_c), sigma_soc(log_l_c))
            return pow(10.0, n)
        else:
            # Case between 8 cm and 11 cm
            n1 = np.random.normal(
                mean_1(self._sat_type, log_l_c), sigma_1(self._sat_type, log_l_c))
            n2 = np.random.normal(
                mean_2(self._sat_type, log_l_c), sigma_2(self._sat_type, log_l_c))
            n = np.random.normal(mean_soc(log_l_c), sigma_soc(log_l_c))

            y1 = pow(10.0, alpha(self._sat_type, log_l_c) * n1 +
                     (1.0 - alpha(self._sat_type, log_l_c)) * n2)
            y0 = pow(10.0, n)

            # beta * y1 + (1 - beta) * y0 = beta * y1 + y0 - beta * y0 = y0 + beta * (y1 - y0)
            return y0 + (characteristic_length - 0.08) * (y1 - y0) / (0.03)

    def _fragment_count(self, min_characteristic_length):
        if self._simulation_type == SimulationType.explosion:
            S = 1
            return int(6 * S * (min_characteristic_length)**(-1.6))
        else:
            print("Computing Count for Collision")

    def _characteristic_length_distribution(self):
        # Sampling a value from uniform distribution
        y = np.random.uniform(0.0, 1.0)
        return power_law(self._min_characteristic_length,
                         self._max_characteristic_length,
                         self._lc_power_law_exponent,
                         y)
