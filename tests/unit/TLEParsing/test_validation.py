import pytest
from odap.Parsing.TLEParser import TLEParser


class TestTLEParserValidation:

    @pytest.mark.parametrize("test_input, expected",
                             [('VANGUARD 1', 'VANGUARD 1'),  # Valid data
                              ('aAgUYHZX7pEr05ybKi5hg8Yqo',
                               ValueError),  # Data too long
                                 ('', ValueError)])  # Data not found
    def test_validate_line0(self, test_input, expected):
        parser = TLEParser()
        if type(expected) == type and issubclass(expected, Exception):
            with pytest.raises(expected):
                parser._TLEParser__validate_line0(test_input)
        else:
            assert parser._TLEParser__validate_line0(test_input) == expected

    # TODO: Finish writing validation tests
    def test_validate_line1(self):
        pass

    def test_validate_line2(self):
        pass
