from abc import ABC, abstractmethod


class Miner(ABC):
    years = None
    legislatures = None

    def __init__(self, years=None, legislatures=None):
        self.years = years
        self.legislatures = legislatures

    @abstractmethod
    def save2CSV(self):
        pass

    @abstractmethod
    def createDataframe(self):
        pass

    @abstractmethod
    def mineData(self):
        pass

    def setYears(self, years):
        self.years = years
    
    def setLegislatures(self, legislatures):
        self.legislatures = legislatures
