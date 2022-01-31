from .ITLEParser import ITLEParser


class TLEParser(ITLEParser):

    # The name provided in a TLE when some of the data may be missing
    __incomplete_tle_name = "TBA - TO BE ASSIGNED"

    """
    Specifications about the data to be parsed from the first line
    of the TLE. The keys respond to the expected location of the data
    in the returned tuple. The value is a tuple where the first value
    is the length of the data and the second value is the name of the
    parsed field.
    """
    __line1_info = {
        0: (5, "num"),
        1: (1, "classification"),
        2: (2, "year"),
        3: (3, "launch"),
        4: (3, "piece"),
        5: (2, "epoch_year"),
        6: (12, "epoch_day"),
        7: (10, "ballistic_coefficient"),
        8: (8, "mean_motion_dotdot"),
        9: (8, "bstar"),
        10: (1, "ephemeris_type"),
        11: (4, "element_number"),
    }

    """
    Specifications about the data to be parsed from the first line
    of the TLE. The keys respond to the expected location of the data
    in the returned tuple. The value is a tuple where the first value
    is the length of the data and the second value is the name of the
    parsed field.
    """
    __line2_info = {
        0: (8, "inclination"),
        1: (8, "raan"),
        2: (7 + 2, "eccentricity"),  # Adding "0." to data for formatting
        3: (8, "aop"),
        4: (8, "mean_anomaly"),
        5: (11, "mean_motion"),
        6: (5, "epoch_rev"),
    }

    def __parse_line0(self, line):
        name = line[2:]
        return name

    def __parse_line1(self, line):
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

    def __parse_line2(self, line):
        inclination = line[8:16]
        raan = line[17:25]
        eccentricity = "0." + line[26:33]
        aop = line[34:42]
        mean_anomaly = line[43:51]
        mean_motion = line[52:63]
        epoch_rev = line[63:68]

        return (inclination, raan, eccentricity,
                aop, mean_anomaly, mean_motion, epoch_rev)

    def __validate_line0(self, data):
        if len(data) > 24:
            raise ValueError('Parsed satellite name was too long')
        if bool(data and not data.isspace()) is False:
            raise ValueError('No satellite name was found in the TLE')
        return data

    def __validate_line1(self, data, name):
        # Validate that the length of data matches the length specified
        # by the TLE format and that the data was successfuly found

        for i in range(len(data)):
            if bool(data[i] and not data[i].isspace()) is False:

                # Some TLES in the dataset do not have completed information, this is an exception
                # to not require data for any fields for those satellites.
                if not self.__incomplete_tle_name == name:
                    raise ValueError(
                        'No satellite %s was found in the TLE for satellite with name `%s`' %
                        (self.__line1_info[i][1], name))

            if not len(data[i]) == self.__line1_info[i][0]:
                raise ValueError(
                    'The length of the parsed field `%s` is did not match the specified length (%s)' %
                    (self.__line1_info[i][1], self.__line1_info[i][0]))
        return data

    def __validate_line2(self, data, name):
        for i in range(len(data)):
            if bool(data[i] and not data[i].isspace()) is False:

                # Some TLES in the dataset do not have completed information, this is an exception
                # to not require data for any fields for those satellites.
                if not self.__incomplete_tle_name == name:
                    raise ValueError(
                        'No satellite %s was found in the TLE for satellite with name `%s`' %
                        (self.__line2_info[i][1], name))

            if not len(data[i]) == self.__line2_info[i][0]:
                raise ValueError(
                    'The length of the parsed field `%s` is did not match the specified length (%s)' %
                    (self.__line2_info[i][1], self.__line2_info[i][0]))
        return data

    def parse(self, tle):
        line0, line1, line2 = tle[0], tle[1], tle[2]

        line0_extracted = self.__extract(line0, 0)
        line0_validated = self.__validate(line0_extracted, 0)

        line1_extracted = self.__extract(line1, 1)
        line1_validated = self.__validate(line1_extracted, 1, line0_validated)

        line2_extracted = self.__extract(line2, 2)
        line2_validated = self.__validate(line2_extracted, 2, line0_validated)

        return line0_validated, line1_validated, line2_validated

    def __extract(self, line, line_number):
        if line_number == 0:
            return self.__parse_line0(line)
        elif line_number == 1:
            return self.__parse_line1(line)
        elif line_number == 2:
            return self.__parse_line2(line)
        else:
            raise ValueError('An invlaid TLE number was provided')

    def __validate(self, data, line_number, name=None):
        if line_number == 0:
            return self.__validate_line0(data)
        elif line_number == 1:
            return self.__validate_line1(data, name)
        elif line_number == 2:
            return self.__validate_line2(data, name)
        else:
            raise ValueError('An invlaid TLE number was provided')
