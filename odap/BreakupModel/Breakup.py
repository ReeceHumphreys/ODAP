from abc import ABC, abstractmethod


class Breakup(ABC):
    @property
    @abstractmethod
    def lc_power_law_exponent(self):
        """
        Gets the exponents used in the characteristic length power law
        :py:meth:`odap.BreakupModel.FragmentationEvent._characteristic_length_distribution`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def delta_velocity_offset(self):
        """
        Gets the offset factors used in determining the change in velocity for each
        fragment.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def max_characteristic_length(self):
        """
        Gets the largest characteristic length possible for the fragmentation event.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def sat_type(self):
        """
        Gets the satellite type for the fragmentation event.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def input_mass(self):
        """
        Gets the input mass for the fragmentation event.
        """
        raise NotImplementedError

    @abstractmethod
    def fragment_count(self, satellites, min_characteristic_length):
        """
        Determines the number of debris fragments produced by the fragmentation event.
        Noteably, this quantity can change if mass conservation is being enforced.

        Parameters
        ----------
        satellites : np.array([Satellite])
            The satellites involved in the fragmentation event. Maximum of two.
        min_characteristic_length : float
            The smallest characteristic length size we wish to generate.
            Note, the smaller the min_characteristic_length, the longer the simulation
            will take to run and the more debris will be generated.
        
        Returns
        -------
        int
            The number of debris generated.
        """
        raise NotImplementedError
