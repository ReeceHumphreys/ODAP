import configparser
from enum import Enum

class SimulationConfiguration:

    class SimulationType(Enum):
        explosion = "EXPLOSION"
        collision = "COLLISION"

    # Takes a .ini file with simulation configurations
    def __init__(self, filePath):
        parser = configparser.ConfigParser()
        parser.read(filePath)
        self._minimalCharacteristicLength = float(parser.get('simulation', 'minimalCharacteristicLength'))
        self._simulationType = self.SimulationType(parser.get('simulation', 'simulationType'))

    @property
    def minimalCharacteristicLength(self):
        return self._minimalCharacteristicLength

    @property
    def simulationType(self):
        return self._simulationType
