from AbstractMiner import Miner
import pandas as pd
import ast
import sys
import numpy as np
import requests
import os


class AuthorsMiner(Miner):
    download_link = "https://dadosabertos.camara.leg.br/arquivos/proposicoesAutores/csv/proposicoesAutores-{year}.csv"
    output_path = "../data/authors/"
    data = None
    infos = []

    def mineData(self):
        for year in self.years:
            response = requests.get(self.download_link.format(year=year))
            file_name = "proposicoesAutores-{}.csv".format(year)
            path = self.output_path.format(file_name=file_name)
            with open(os.path.join(path, file_name), 'wb') as f:
                f.write(response.content)

    def createDataframe(self):
        for year in self.years:
            proposal_authors = pd.read_csv("../data/authors/proposicoesAutores-{}.csv".format(year), sep=';')
            proposal_authors = proposal_authors[['idProposicao', 'idDeputadoAutor', 'codTipoAutor']]
            proposal_authors = proposal_authors.dropna()
            proposal_authors['idDeputadoAutor'] = proposal_authors['idDeputadoAutor'].astype(int)
            proposal_authors.rename(columns={'idDeputadoAutor': 'idAutor'}, inplace=True)
            self.infos.append(proposal_authors)

    def save2CSV(self):
        data = pd.concat(self.infos)
        data.to_csv('../data/authors_info.csv', header=True, index=False)