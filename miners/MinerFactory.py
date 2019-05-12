import json
from APIProposalMiner import APIProposalMiner
from AuthorsMiner import AuthorsMiner
from DeputiesMiner import DeputiesMiner
from PartiesMiner import PartiesMiner
from ProposalsMiner import ProposalsMiner
from RolesMiner import RolesMiner
from TSEMiner import TSEMiner


class MinerFactory():
    config = None

    def __init__(self, miners, years, legislatures, start_date=None, end_date=None):
        self.years = years
        self.legislatures = legislatures
        self.miners = miners
        self.start_date = start_date
        self.end_date = end_date

    def buildAll(self):
        class_map = {
            "APIProposalMiner": APIProposalMiner(years=self.years, legislatures=self.legislatures),
            "AuthorsMiner": AuthorsMiner(years=self.years, legislatures=self.legislatures),
            "DeputiesMiner": DeputiesMiner(years=self.years, legislatures=self.legislatures),
            "PartiesMiner": PartiesMiner(years=self.years, legislatures=self.legislatures),
            "ProposalsMiner": ProposalsMiner(years=self.years, legislatures=self.legislatures),
            "RolesMiner": RolesMiner(years=self.years, legislatures=self.legislatures),
            "TSEMiner": TSEMiner(years=self.years, legislatures=self.legislatures)
        }
        print(self.miners)
        for miner in self.miners:
            print("Extraindo dados de: {}".format(miner))
            minerInstance = class_map[miner]
            if(miner == 'RolesMiner'):
                minerInstance.setDates(self.start_date, self.end_date)
            minerInstance.mineData()
            minerInstance.createDataframe()
            minerInstance.save2CSV()