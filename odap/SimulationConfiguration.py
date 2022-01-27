import configparser
from enum import Enum
class SimulationType(Enum):
        explosion = "EXPLOSION"
        collision = "COLLISION"
class SimulationConfiguration:

    # Takes a .ini file with simulation configurations
    def __init__(self, filePath):
        parser = configparser.ConfigParser()
        parser.read(filePath)
        self._minimalCharacteristicLength = float(parser.get('simulation', 'minimalCharacteristicLength'))
        self._simulationType = SimulationType(parser.get('simulation', 'simulationType'))

    @property
    def minimalCharacteristicLength(self):
        return self._minimalCharacteristicLength

    @property
    def simulationType(self):
        return self._simulationType
