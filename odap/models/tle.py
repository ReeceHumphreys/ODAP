from ..utils import _M_to_nu
import numpy as np

# unit: [km^3 / s^2]
mu_Earth = 3.986004418e5


class TLE:

    """
    Parse a TLE from its component line
    `name` is the "0th" line of the TLE
    """

    def __init__(self, name, line1, line2):
        self.name = name

        # Parsing Line 1
        self.norad = line1[2:7].strip()
        self.classification = line1[7]
        self.int_desig = line1[9:17].strip()
        self.epoch_year = line1[18:20].strip()
        self.epoch_day = line1[20:32].strip()
        self.ballistic_coef = float(line1[33:43])
        self.dd_n = _parse_float(line1[44:52])
        self.bstar = _parse_float(line1[53:61])
        self.set_num = line1[64:68].strip()

        # Parsing Line 2
        self.inc = float(line2[8:16])  # unit: [deg]
        self.raan = float(line2[17:25])  # unit: [deg]
        self.ecc = _parse_decimal(line2[26:33])  # unit: []
        self.argp = float(line2[34:42])  # unit: [deg]
        self.M = float(line2[43:51])  # unit: [deg]
        self.n = float(line2[52:63])  # unit: [Revs per day]
        self.rev_num = line2[63:68].strip()  # unit: [Revs]

        # Setting properties that need computing to `None`
        self._epoch = None
        self._a = None
        self._nu = None

    @property
    def epoch(self):
        """Epoch of the TLE."""
        if self._epoch is None:
            year = np.datetime64(self.epoch_year - 1970, "Y")
            day = np.timedelta64(
                int((self.epoch_day - 1) * 86400 * 10 ** 6), "us"
            )
            self._epoch = year + day
        return self._epoch


    @property
    def a(self):
        """
        Semi-major axis.

        Returns
        -------
        float
            The semi-major axis of the satellite specified by the tle. Unit: [km]
        """
        if self._a is None:
            self._a = mu_Earth**(1/3) / ((2 * self.n * np.pi)/ 86400)**(2/3)
        return self._a

    @property
    def nu(self):
        """
        True anomaly.

        Returns
        -------
        float
            The true anomaly of the satellite specified by the tle. Unit: [deg]
        """
        if self._nu is None:
            # Wrap M to [-pi, pi]
            M = (np.deg2rad(self.M) + np.pi) % (2 * np.pi) - np.pi
            ecc = self.ecc
            nu_rad = (_M_to_nu(M, ecc) + np.pi) % (2 * np.pi) - np.pi
            # Wrap nu t0 [0, 360]
            self._nu = np.rad2deg(nu_rad) % 360
        return self._nu


def _conv_year(s):
    """Interpret a two-digit year string."""
    if isinstance(s, int):
        return s
    y = int(s)
    return y + (1900 if y >= 57 else 2000)


def _parse_decimal(s):
    """Parse a floating point with implicit leading dot.
    >>> _parse_decimal('378')
    0.378
    """
    return float("." + s)


def _parse_float(s):
    """Parse a floating point with implicit dot and exponential notation.
    >>> _parse_float(' 12345-3')
    0.00012345
    """
    return float(s[0] + "." + s[1:6] + "e" + s[6:8])
