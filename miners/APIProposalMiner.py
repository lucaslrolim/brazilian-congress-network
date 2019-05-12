# DEPEDRATED #

# Authors info can be easly obteined acessing https://dadosabertos.camara.leg.br/swagger/api.html#staticfile
# the snippet below works, but takes a lot of time

from AbstractMiner import Miner
import http.client
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import sys
import ast
from bs4 import BeautifulSoup
from utils import addProposalType
from utils import addProposalYear
from utils import addProposalSituation
from utils import addStatus
from utils import printProgressBar


class APIProposalMiner(Miner):
    proposal_types = ["PL", "PEC", "PLN", "PLP", "PLV", "PLC"]
    # Resquests models to House of Representatives` API v 2.0
    api_author_info = "https://dadosabertos.camara.leg.br/api/v2/proposicoes/{proposal_code}/autores?formato=json"
    api_votes_info = "https://dadosabertos.camara.leg.br/api/v2/proposicoes/{proposal_code}/votacoes?formato=json"
    api_proposal_info = "https://dadosabertos.camara.leg.br/api/v2/proposicoes/{proposal_code}?formato=json"
    api_url = "https://dadosabertos.camara.leg.br/api/v2/proposicoes?&pagina={page_number}&itens=100&ordem=DESC&ordenarPor=ano&formato=json"

    output_path = "../data/"
    proposals = {}
    data = None

    def mineData(self):
        self.loadProposals()
        self.loadProposalsInfo()

    def createDataframe(self):
        self.data = pd.DataFrame.from_dict(self.proposals, orient='index').reset_index()

    def save2CSV(self):
        self.data.to_csv(self.output_path + 'proposals_info.csv', header=True, index=False)

    def setProposalTypes(self, proposal_types):
        self.proposal_types = proposal_types

    def getProposalTypes(self):
        return self.proposal_types

    def loadProposals(self):
        # Add filters to API request
        api_url = addProposalYear(self.api_url, self.years)
        api_url = addProposalType(self.api_url, self.proposal_types)
        api_url = addStatus(self.api_url, True)
        page_number = 1
        # Get proposals according to the filter defined
        while True:
            response_basic_proposal_info = requests.get(self.api_url.format(page_number=page_number))
            print("Reading {} result page of basic proposal ids".format(page_number))
            if response_basic_proposal_info.status_code == 200:
                try:
                    response = response_basic_proposal_info.json()['dados']
                    for proposal_item in response:
                        proposal_id = proposal_item['id']
                        proposal_type = proposal_item['siglaTipo']
                        self.proposals[proposal_id] = {'type': proposal_type}
                    response_basic_proposal_info.json()['links'][3]['href']
                except:
                    break
                page_number += 1
            else:
                print("error getting proposal ids")

    def loadProposalsInfo(self):
        proposals_ids = list(self.proposals.keys())
        printProgressBar(0, len(proposals_ids), prefix='Info about proposal authors:', suffix='Complete', length=50)
        progress = 0
        # Get info about proposals´s authors and subject
        for proposal_code in proposals_ids:
            # Basic proposal info
            response_proposal_info = requests.get(self.api_proposal_info.format(proposal_code=proposal_code))
            if response_proposal_info.status_code == 200:
                try:
                    response_proposal_info = response_proposal_info.json()['dados']
                    self.proposals[proposal_code]['number'] = response_proposal_info['numero']
                    self.proposals[proposal_code]['year'] = response_proposal_info['ano']
                    self.proposals[proposal_code]['proceduring'] = response_proposal_info['statusProposicao']['idTipoTramitacao']
                    self.proposals[proposal_code]['subject'] = response_proposal_info['ementa']
                    self.proposals[proposal_code]['keywords'] = response_proposal_info['keywords']
                    # Author's info
                    response_author_info = requests.get(self.api_author_info.format(proposal_code=proposal_code))
                    if response_author_info.status_code == 200:
                        try:
                            response_author_info = response_author_info.json()['dados']
                            authors_ids = []
                            for element in response_author_info:
                                uri = element['uri']
                                if uri is not None:
                                    # Get author id from author profile url
                                    deputie_id = uri.rsplit('/', 1)[-1]
                                    authors_ids.append(deputie_id)
                                else:
                                    authors_ids.append(element['nome'])
                            proposals[proposal_code]['authors'] = authors_ids
                        except:
                            print("error reading authors JSON: ", proposal_code)
                    else:
                        print("error loading proposal authors API: ", proposal_code)

                except:
                    print("error getting information from proposal JSON: ", proposal_code)
            else:
                print("error loading proposal basic info API: ", proposal_code)
            progress += 1
            printProgressBar(progress, len(proposals_ids), prefix='Info about proposal authors:', suffix='Complete', length=50)