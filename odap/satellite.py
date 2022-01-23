class Satellite:

    __incomplete_tle_name = "TBA - TO BE ASSIGNED"

    __line1_data_lengths = {
        0: 5,
        1: 1,
        2: 2,
        3: 3,
        4: 3,
        5: 2,
        6: 12,
        7: 10,
        8: 8,
        9: 8,
        10: 1,
        11: 4,
        }

    __line1_data_names = {
        0: "num",
        1: "classification",
        2: "year",
        3: "launch",
        4: "piece",
        5: "epoch_year",
        6: "epoch_day",
        7: "ballistic_coefficient",
        8: "mean_motion_dotdot",
        9: "bstar",
        10: "ephemeris_type",
        11: "element_number",
        }

    __line2_data_lengths = {
        0: 8,
        1: 8,
        2: 7,
        3: 8,
        4: 8,
        5: 11,
        6: 5,
        }

    __line2_data_names = {
        0: "inclination",
        1: "raan",
        2: "eccentricity",
        3: "aop",
        4: "mean_anomaly",
        5: "mean_motion",
        6: "epoch_rev",
        }

    def __tle_parser(self, line, line_number):
        if line_number == 0:
            return self.__tle_parser_zero(line)
        elif line_number == 1:
            return self.__tle_parser_one(line)
        elif line_number == 2:
            return self.__tle_parser_two(line)
        else:
            raise ValueError('An invlaid TLE number was provided')

    def __tle_parser_zero(self, line):
        return line[2:]

    def __tle_parser_one(self, line):
        num = line[2:7]
        classification = line[7]
        year = line[9:11]
        launch = line[11:14]
        piece = line[14:17]
        epoch_year = line[18:20]
        epoch_day = line[20:32]
        ballistic_coefficient = line[33:43]
        mean_motion_dotdot = line[44:52]
        bstar = line[53:61]
        ephemeris_type = line[62]
        element_number = line[65:69]

        return (num, classification, year, launch,
        piece, epoch_year, epoch_day, ballistic_coefficient,
        mean_motion_dotdot, bstar, ephemeris_type, element_number)

    def __tle_parser_two(self, line):
        inclination = line[8:16]
        raan = line[17:25]
        eccentricity = line[26:33]
        aop = line[34:42]
        mean_anomaly = line[43:51]
        mean_motion = line[52:63]
        epoch_rev   = line[63:68]

        return (inclination, raan, eccentricity, 
        aop, mean_anomaly, mean_motion, epoch_rev)

    def __tle_validator(self, data, line_number):
        if line_number == 0:
            return self.__tle_validator_zero(data)
        elif line_number == 1:
            return self.__tle_validator_one(data)
        elif line_number == 2:
            return self.__tle_validator_two(data)
        else:
            raise ValueError('An invlaid TLE number was provided')
    
    def __tle_validator_zero(self, data):
        if len(data) > 24: raise ValueError('Parsed satellite name was too long')
        if bool(data and not data.isspace()) == False: raise ValueError('No satellite name was found in the TLE')
        return data

    def __tle_validator_one(self, data):
        # Validate that the length of data matches the length specified
        # by the TLE format and that the data was successfuly found

        for i in range(len(data)):
            if bool(data[i] and not data[i].isspace()) == False:
                # Some TLES in the dataset do not have completed information, this is an exception 
                # to not require data for any fields for those satellites.

                # TODO: Currently taking advantage of knowing we parsed line0 before line1. However it is not 
                # necessary to do it in this order thus using `self.name` should be avoided.
                if not self.__incomplete_tle_name == self.name:
                    raise ValueError('No satellite %s was found in the TLE for satellite with name `%s`' % (self.__line1_data_names[i], self.name))

            if not len(data[i]) == self.__line1_data_lengths[i]:
                raise ValueError('The length of the parsed field `%s` is did not match the specified length (%s)' % (self.__line1_data_names[i], self.__line1_data_names[i]))
        return data
    
    def __tle_validator_two(self, data):
        for i in range(len(data)):
            if bool(data[i] and not data[i].isspace()) == False:
                # Some TLES in the dataset do not have completed information, this is an exception 
                # to not require data for any fields for those satellites.

                # TODO: Currently taking advantage of knowing we parsed line0 before line1. However it is not 
                # necessary to do it in this order thus using `self.name` should be avoided.
                if not self.__incomplete_tle_name == self.name:
                    raise ValueError('No satellite %s was found in the TLE for satellite with name `%s`' % (self.__line2_data_names[i], self.name))

            if not len(data[i]) == self.__line2_data_lengths[i]:
                raise ValueError('The length of the parsed field `%s` is did not match the specified length (%s)' % (self.__line2_data_names[i], self.__line2_data_names[i]))
        return data


    def __init__(self, tle):
        """
        Constructs all the necessary attributes for the satellite object.

        Parameters
        ----------
            tles : array 
                an (3, ) NumPy array where each element corresponds to the line of a two-line element

        """
        # Need to slice tle line 1 as each name begins with "0 "
        line0, line1, line2 = tle[0], tle[1], tle[2]

        # Extracting data from the first line of the tle
        line0_parsed = self.__tle_parser(line0, 0)
        line0_validated = self.__tle_validator(line0_parsed, 0)
        self.name = line0_validated

        # Extracting data from the second line of the tle
        line1_parsed = self.__tle_parser(line1, 1)
        line1_validated = self.__tle_validator(line1_parsed, 1)

        (self.num, self.classification, self.year, self.launch, self.piece,
        self.epoch_year, self.epoch_day, self.ballistic_coefficient,
        self.mean_motion_dotdot, self.bstar, self.ephemeris_type, self.element_number) = line1_validated

        # Extracting data from the third line of the tle
        line2_parsed = self.__tle_parser(line2, 2)
        line2_validated = self.__tle_validator(line2_parsed, 2)
        (self.inclination, self.raan, self.eccentricity, self.aop,
         self.mean_anomaly, self.mean_motion, self.epoch_rev) = line2_validated