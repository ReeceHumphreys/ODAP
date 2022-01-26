import configparser

class ConfigurationParser:


    def __init__(self):
        self.parser = configparser.ConfigParser()

    def parse(self, filePath):
        self.parser.read(filePath)
        sections = self.parser.sections()
        print(sections)

