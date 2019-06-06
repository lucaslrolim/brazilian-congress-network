from .AbstractMiner import Miner
import pandas as pd
import requests
import csv
from .utils import addLegislature


class PartiesMiner(Miner):
    api_parties_info = "https://dadosabertos.camara.leg.br/api/v2/partidos/{party_id}?formato=json"
    api_parties_list = "https://dadosabertos.camara.leg.br/api/v2/partidos?&pagina={page_number}&itens=100&ordem=ASC&ordenarPor=sigla&formato=json"

    output_path = "../data/"
    parties = {}
    data = None

    def mineData(self):
        self.loadLegislaturesPartiesData()
        self.loadPartiesInfos()

    def createDataframe(self):
        self.data = (pd.DataFrame.from_dict(self.parties, orient='index').reset_index())

    def save2CSV(self):
        self.data.to_csv(self.output_path + 'parties_info.csv', header=True, index=False)

    def loadLegislaturesPartiesData(self):
        api_parties_legislature = addLegislature(self.api_parties_list, self.legislatures)
        page_number = 1
        while True:
            response_basic_proposal_info = requests.get(api_parties_legislature.format(page_number=page_number))
            if response_basic_proposal_info.status_code == 200:
                try:
                    response = response_basic_proposal_info.json()['dados']
                    for item in response:  
                        party_id = item['id']
                        party_initials = item['sigla']
                        party_name = item['nome']
                        self.parties[party_id] = {'name': party_name, 'initials': party_initials}
                    # request to check if there is one more page
                    response_basic_proposal_info.json()['links'][3]['href']
                except:
                    break
                page_number += 1
            else:
                print("error getting proposal ids")

    def loadPartiesInfos(self):
        parties_ids_list = list(self.parties.keys())

        for party_id in parties_ids_list:
            response = requests.get(self.api_parties_info.format(party_id=party_id))
            if response.status_code == 200:
                try:
                    response = response.json()['dados']
                    party_status = response['status']['situacao']
                    members_number = response['status']['totalMembros']
                    if party_status != "Inativo" and int(members_number) != 0:
                        uri = response['status']['lider']["uri"]
                        leader_id = uri.rsplit('/', 1)[-1]
                        leader_name = response['status']['lider']["nome"]
                        self.parties[party_id]['leader_name'] = leader_name
                        self.parties[party_id]['leader_id'] = leader_id
                        self.parties[party_id]['members_number'] = members_number
                    else:
                        self.parties.pop(party_id, None)
                except:
                    print("json error", party_id)
            else:
                print("error: ", party_id)