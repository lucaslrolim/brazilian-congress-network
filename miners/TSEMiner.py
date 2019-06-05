from AbstractMiner import Miner
import pandas as pd
import ast
import sys
import requests
import os
import zipfile
from utils import printProgressBar

class TSEMiner(Miner):
    proposal_types = ["PL", "PEC", "PLN", "PLP", "PLV", "PLC"]
    infos = []
    output_path = "../data/candidates/"
    output_zip_path = "../data/candidates/temp/"
    main_data_path = "../data/"
    valid_years = [2014, 2018, 2022]
    election_years = []
    data = None
    download_link = "http://agencia.tse.jus.br/estatistica/sead/odsele/consulta_cand/consulta_cand_{year}.zip"

    def __init__(self, **kwargs):
        super(TSEMiner, self).__init__(**kwargs)
        for year in self.years:
            if year in self.valid_years:
                self.election_years.append(year)

    def mineData(self):
        self.dowloadZip()
        self.extractFile()

    def createDataframe(self):
        for year in self.election_years:
                candidates = pd.read_csv("../data/candidates/consulta_cand_{}_BRASIL.csv".format(year), sep=';', encoding='latin1', low_memory=False)
                candidates = candidates[['NM_CANDIDATO', 'NR_CPF_CANDIDATO', 'SG_PARTIDO', 'NM_COLIGACAO', 'SG_UF_NASCIMENTO', 'DS_GRAU_INSTRUCAO', 'DS_COR_RACA']]
                candidates.replace('"', '')
                self.infos.append(candidates)
        self.data = pd.concat(self.infos)

    def save2CSV(self):
        self.data.to_csv(self.main_data_path + 'candidates_tse_info.csv', header=True, index=False)

    def dowloadZip(self):
        printProgressBar(0, len(self.years), prefix='Fazendo download de arquivos .zip:', suffix='Complete', length=50)
        progress = 0
        for year in self.election_years:
            response = requests.get(self.download_link.format(year=year))
            file_name = "consulta_cand_{year}.zip".format(year=year)
            path = self.output_path.format(file_name=file_name)

            if not os.path.exists(path):
                os.makedirs(path)

            path = self.output_path.format(file_name=file_name)
            with open(os.path.join(self.output_zip_path, file_name), 'wb') as f:
                f.write(response.content)
                f.close()
        pass

    def extractFile(self):
        for year in self.election_years:
            file_name = "consulta_cand_{year}.zip".format(year=year)
            with zipfile.ZipFile(self.output_zip_path + file_name, "r") as zip_ref:
                zip_ref.extract(path=self.output_path, member="consulta_cand_{year}_BRASIL.csv".format(year=year))