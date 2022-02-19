from odap.BreakupModel.Breakup import Breakup


class Explosion(Breakup):
    @property
    def lc_power_law_exponent(self):
        """
        Gets the exponents used in the characteristic length power law
        :py:meth:`odap.BreakupModel.FragmentationEvent._characteristic_length_distribution`.
        """
        return -2.6

    @property
    def delta_velocity_offset(self):
        """
        Gets the offset factors used in determining the change in velocity for each
        fragment.
        """
        return [0.2, 1.85]

    @property
    def max_characteristic_length(self):
        """
        Gets the largest characteristic length possible for the fragmentation event.
        For explosions, this is the characteristic length of the input satellite.
        """
        return self._max_characteristic_length

    @property
    def sat_type(self):
        """
        Gets the satellite type for the fragmentation event.
        """
        return self._sat_type

    @property
    def input_mass(self):
        """
        Gets the input mass for the fragmentation event. For explosions, this is the
        mass of the input satellite.
        """
        return self._input_mass

    def fragment_count(self, satellites, min_characteristic_length):
        """
        Determines the number of debris fragments produced by the fragmentation event.
        Noteably, this quantity can change if mass conservation is being enforced.

        Parameters
        ----------
        satellites : np.array([Satellite])
            The satellites involved in the fragmentation event. Maximum of one.
        min_characteristic_length : float
            The smallest characteristic length size we wish to generate.
            Note, the smaller the min_characteristic_length, the longer the simulation
            will take to run and the more debris will be generated.
        
        Returns
        -------
        int
            The number of debris generated.
        """
        satellite = satellites[0]
        self._max_characteristic_length = satellite.characteristic_length
        self._sat_type = satellite.type
        self._input_mass = satellite.mass

        S = 1
        return int(6 * S * (min_characteristic_length) ** (-1.6))

