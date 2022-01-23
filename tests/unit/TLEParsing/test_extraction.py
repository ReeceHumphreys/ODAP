import pytest
from odap.Parsing.TLEParser import TLEParser

class TestTLEParserExtraction:

    @pytest.mark.parametrize("test_input, expected",
    [('0 VANGUARD 1', 'VANGUARD 1')])
    def test_parse_line0(self, test_input, expected):
        parser = TLEParser()
        parsed_line0 = parser._TLEParser__parse_line0(test_input)

        assert parsed_line0 == expected

    @pytest.mark.parametrize("test_input, expected",
    [
        ('1 00005U 58002B   22022.55771106  .00000105  00000-0  15609-3 0  9990', #input
        ('00005', 'U', '58', '002', 'B  ', '22', '022.55771106', ' .00000105', ' 00000-0', ' 15609-3', '0', '9990')), #expected
    ])
    def test_parse_line1(self, test_input, expected):
        parser = TLEParser()
        parsed_line0 = parser._TLEParser__parse_line1(test_input)

        assert parsed_line0 == expected

    @pytest.mark.parametrize("test_input, expected",
    [
        ('2 00005  34.2401 228.9165 1843365 100.8944 280.3452 10.84849428268693', #input
        (' 34.2401', '228.9165', '1843365', '100.8944', '280.3452', '10.84849428', '26869')), #expected
    ])
    def test_parse_line2(self, test_input, expected):
        parser = TLEParser()
        parsed_line0 = parser._TLEParser__parse_line2(test_input)

        assert parsed_line0 == expected

    @pytest.mark.parametrize("test_input, expected",
    [
        (('0 VANGUARD 1', '1 00005U 58002B   22022.55771106  .00000105  00000-0  15609-3 0  9990', '2 00005  34.2401 228.9165 1843365 100.8944 280.3452 10.84849428268693'), #input
        ('VANGUARD 1', ('00005', 'U', '58', '002', 'B  ', '22', '022.55771106', ' .00000105', ' 00000-0', ' 15609-3', '0', '9990'), (' 34.2401', '228.9165', '1843365', '100.8944', '280.3452', '10.84849428', '26869'))), #expected
    ])
    def test_parse(self, test_input, expected):
        parser = TLEParser()
        parsed_line0 = parser.parse(test_input)

        assert parsed_line0 == expected








