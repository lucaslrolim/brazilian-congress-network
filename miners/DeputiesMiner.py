from AbstractMiner import Miner

import pandas as pd
import requests
from utils import printProgressBar


class DeputiesMiner(Miner):
    api_legislature_info = "https://dadosabertos.camara.leg.br/api/v2/deputados?idLegislatura={legislature_number}&pagina={page_number}&itens=100&ordem=ASC&ordenarPor=nome&formato=json"
    api_deputy_info = "https://dadosabertos.camara.leg.br/api/v2/deputados/{deputy_id}?formato=json"
    output_path = "../data/"
    deputies = {}
    data = None

    def mineData(self):
        self.loadLegislatureDeputies()
        self.loadDeputiesInfo()

    def createDataframe(self):
        self.data = pd.DataFrame.from_dict(self.deputies, orient='index').reset_index()

    def save2CSV(self):
        self.data.to_csv(self.output_path + 'deputies_info.csv', header=True, index=False)

    def loadLegislatureDeputies(self):
        for legislature in self.legislatures:
            page_number = 1
            while page_number <= 7:
                response = requests.get(self.api_legislature_info.format(
                    legislature_number=legislature, page_number=page_number)
                )
                if response.status_code == 200:
                    try:
                        response = response.json()['dados']
                        for deputie in response:
                            party = deputie['siglaPartido']
                            uf = deputie['siglaUf']
                            name = deputie['nome']
                            url_photo = deputie['urlFoto']
                            self.deputies[deputie['id']] = {'party': party, 'uf': uf, 'name': name, 'photo':url_photo}
                    except:
                        print("json error {} on page {} and id {}".format(name, page_number, deputie))
                else:
                    print("error: ", page_number)
                page_number += 1
            print("{} deputies mined on legislature {}".format(len(self.deputies), legislature))

    def loadDeputiesInfo(self):
        ids_list = list(self.deputies.keys())
        printProgressBar(0, len(ids_list), prefix='Detailed info about deputies:', suffix='Complete', length=50)
        progress = 0
        for deputy_id in ids_list:
            response = requests.get(self.api_deputy_info.format(deputy_id=deputy_id))
            if response.status_code == 200:
                try:
                    response = response.json()['dados']
                    self.deputies[deputy_id]['sex'] = response['sexo']
                    self.deputies[deputy_id]['education'] = response['escolaridade']
                    self.deputies[deputy_id]['hometown'] = response['municipioNascimento']
                    self.deputies[deputy_id]['birthdate'] = response['dataNascimento']
                    self.deputies[deputy_id]['cpf'] = response['cpf']
                except Exception as e:
                    print(e)
                    print("Error on json deputy: ", deputy_id) 
            else:
                print("Error on deputy: ", deputy_id)
            progress += 1
            printProgressBar(progress, len(ids_list), prefix='Detailed info about deputies:', suffix='Complete', length=50)