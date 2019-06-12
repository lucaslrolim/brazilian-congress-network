from .AbstractMiner import Miner
import pandas as pd
import ast
import sys
import requests
import os


class ProposalsMiner(Miner):
    proposal_types = ["PL", "PEC", "PLN", "PLP", "PLV", "PLC"]
    infos = []
    download_link = "https://dadosabertos.camara.leg.br/arquivos/proposicoes/csv/proposicoes-{year}.csv"
    output_path = "../data/proposals/"

    def mineData(self):
        for year in self.years:
            response = requests.get(self.download_link.format(year=year))
            file_name = "proposicoes-{}.csv".format(year)
            path = self.output_path.format(file_name=file_name)
            with open(os.path.join(path, file_name), 'wb') as f:
                f.write(response.content)

    def createDataframe(self):
        for year in self.years:
            proposal = pd.read_csv("../data/proposals/proposicoes-{}.csv".format(year), sep=';')
            proposal = proposal[['id', 'ultimoStatus_idSituacao', 'siglaTipo', 'ano', 'ementa', 'keywords']]
            proposal = proposal[proposal['siglaTipo'].isin(self.proposal_types)]
            proposal['ultimoStatus_idSituacao'] = proposal['ultimoStatus_idSituacao'].fillna(0.0).astype(int)
            self.infos.append(proposal)

    def save2CSV(self):
        data = pd.concat(self.infos)
        data.to_csv('../data/proposals_info.csv', header=True, index=False)

    def setProposalTypes(self, proposal_types):
        self.proposal_types = proposal_types

    def getProposalTypes(self):
        return self.proposal_types