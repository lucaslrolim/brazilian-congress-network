from .AbstractMiner import Miner
import pandas as pd
import requests
import csv
import os


class RolesMiner(Miner):
    api_groups_info = "https://dadosabertos.camara.leg.br/api/v2/deputados/{deputy_id}/orgaos?dataInicio={data_inicio}&dataFim={data_fim}&itens=100&ordem=ASC&formato=json"
    api_mesa_info = "https://dadosabertos.camara.leg.br/api/v2/legislaturas/{legislature}/mesa?formato=json"
    download_link = "https://dadosabertos.camara.leg.br/arquivos/proposicoes/csv/proposicoes-{year}.csv"
    output_path = "../data/roles/"
    dates = {}
    deputies_list = []
    col_names = ['deputy_id', 'role_name', 'role_place_id', 'role_place_name']
    df = None

    def __init__(self, start_date=None, end_date=None, col_names=None, **kwargs):
        super(RolesMiner, self).__init__(**kwargs)
        self.dates['start'] = start_date
        self.dates['end'] = end_date
        if(col_names is not None):
            self.col_names = col_names

        self.df = pd.DataFrame(columns=self.col_names)

    def mineData(self):
        self.setDeputiesList()
        self.loadDeputiesGroups()
        self.loadDeputiesPartyRole()

    def createDataframe(self):
        pass

    def save2CSV(self):
        self.df.to_csv('../data/roles_info.csv', header=True, index = False)

    def getDeputiesList(self):
        return self.deputies_list

    def setDeputiesList(self):
        with open('../data/deputies_info.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.deputies_list.append(row['index'])

    def loadDeputiesGroups(self):
        for deputy_id in self.deputies_list:
            response = requests.get(self.api_groups_info.format(
                data_inicio=self.dates['start'], data_fim=self.dates['end'], deputy_id=deputy_id)
            )
            if response.status_code == 200:
                try:
                    response = response.json()['dados']
                    for group in response:
                        self.df.loc[len(self.df)] = [deputy_id, group['titulo'], group['idOrgao'], group['nomeOrgao']]
                except Exception as ex:
                    print("json error deputy: ", deputy_id)
                    print(ex)
            else:
                print("error: ", deputy_id)

        for legislature in self.legislatures:
            response = requests.get(self.api_mesa_info.format(legislature=legislature))
            if response.status_code == 200:
                try:
                    response = response.json()['dados']
                    for deputy in response:
                        self.df.loc[len(self.df)] = [deputy['id'], deputy['nomePapel'], None, None]
                except:
                    print("json error", deputy_id)
            else:
                print("error: ", deputy_id)
    
    def loadDeputiesPartyRole(self):
        with open('../data/parties_info.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            role_model = "Líder do {party_name}"
            for row in reader:
                deputy_id = row['leader_id']
                party_name = row['name']
                party_id = row['index']
                role_name = role_model.format(party_name=party_name)
                self.df.loc[len(self.df)] = [deputy_id, role_name, party_id, party_name]

    def setDates(self, start, end):
        self.dates['start'] = start
        self.dates['end'] = end

    def setColnames(self, col_names):
        self.col_names = col_names