import numpy as np
import networkx as nx
import pydot
import random
import ast
from .model_parameters import *
from .data_readers import *
import random
import ast
from datetime import date
from .utils import calculateAge
from .utils import getAgeRange
from .utils import generateEdges
from .utils import getUfRegion

class NetworkBuilder():
    deputies = None
    deputies_ids = None
    proposals = None
    legislative_roles = None
    parties = None
    proposal_authors = None
    tse_info = None

    deputies_proposals = None
    deputies_proposals_pertinence = None
    roles_relevance = None

    G = None
    collab_weights = {}
    collab_pertinence = {}

    def __init__(self):
        print("Carregando informações...")
        self.deputies = getDeputies()
        self.deputies_ids = list(self.deputies.keys())
        self.proposals = getProposals()
        self.legislative_roles = getRoles()
        self.parties = getParties()
        self.proposal_authors = getAuthors()
        self.tse_info = getInfoTSE()
        self.setDeputiesRegion()
        self.setDeputiesIndividualProposals()
        self.setDeputiesRoleInfluence()

    def buildNetwork(self, weighted = True):
        self.weighted_network = weighted
        self.G = nx.Graph()
        self.addNodes()
        self.addEdges()
        self.removePastDeputies()

    def saveNetWork(self, network_name="coauthorship-network", use_version=True):
        print("Salvando a rede...")
        if(use_version):
            nx.write_gexf(self.G, "../data/networks/{}-{}.gexf".format(network_name, date.today()))
        else:
            nx.write_gexf(self.G, "../data/networks/{}.gexf".format(network_name))
        print("Rede salva em: {}".format("../data/networks/"))

    def addNodes(self):
        print("Gerando vértices...")
        out_of_date_deputies = []

        for deputy_id in self.deputies_ids:
            if (int(deputy_id) in list(self.deputies.keys())):
                deputy_id = int(deputy_id)
                cpf = self.deputies[deputy_id]['cpf']
                party = self.deputies[deputy_id]['party']
                uf = self.deputies[deputy_id]['uf']
                region = self.deputies[deputy_id]['region']
                label = self.deputies[deputy_id]['name']
                sex = self.deputies[deputy_id]['sex']
                education = self.deputies[deputy_id]['education']
                education_tse = self.tse_info[cpf]['DS_GRAU_INSTRUCAO']
                ethnicity = self.tse_info[cpf]['DS_COR_RACA']
                age = calculateAge(self.deputies[deputy_id]['birthdate'])
                age_range = getAgeRange(age)

                if (deputy_id in self.deputies_proposals):
                    individual_proposals = self.deputies_proposals[deputy_id] * node_parameters['proposal']
                else:
                    individual_proposals = 0

                if (deputy_id in self.roles_relevance):
                    role_relevance = self.roles_relevance[deputy_id] * node_parameters['role']
                else:
                    role_relevance = 0
                deputy_weight = individual_proposals + role_relevance
                self.G.add_node(
                    deputy_id, label=label, style='filled', weight=deputy_weight, party=party, uf=uf,
                    age_range=age_range, sex=sex, education=education, age=age,
                    education_tse=education_tse, ethnicity=ethnicity, region=region
                    )
            else:
                out_of_date_deputies.append(deputy_id)

    def addEdges(self):
        print("Adicionando arestas...")
        self.setCollaborations()
        self.setCollaborationsSuccess()
        for edge in self.collab_weights.keys():
            if(self.weighted_network):
                weight = self.collab_weights[edge]
            else:
                weight = 1
            self.G.add_edge(
                edge[0], edge[1], weight= weight, success_pertinence=self.collab_pertinence[edge]
            )
    
    def removePastDeputies(self):
        for deputy in self.G.nodes():
            if(deputy not in self.deputies_ids):
                self.G.remove_node(deputy)

    def setCollaborations(self):
        '''
        Define pesos das arestas de acordo com número de coautoria entre dois deputados
        '''
        for proposal_id in list(self.proposals.keys()):
            if proposal_id in list(self.proposal_authors.keys()):
                proposal_authors = self.proposal_authors[proposal_id]
                proposal_type = self.proposals[proposal_id]['siglaTipo']
                n_proposal_weight = proposal_weight[proposal_type]
                # proposals with only one author add node weight and will not be considered agai
                if(len(proposal_authors) > 1):
                    self.collab_weights = self.addCollabEdge(
                        self.collab_weights, proposal_authors, n_proposal_weight, False
                    )

    def addCollabEdge(self, graph, collab_list, proposal_weight, archived):
        '''
        Calcula o número de colaborações que ocorreu entre os deputados em uma proposicao
        '''
        proposal_weight = proposal_weight
        if bool(archived) is True:
            proposal_weight = proposal_weight * 0.5

        edges = generateEdges(collab_list)
        for edge in edges:
            if edge in graph:
                graph[edge] += proposal_weight
            else:
                graph[edge] = proposal_weight

        return graph

    def addCollabPertinence(self, graph, collab_list, proposal_pertinence):
        '''
        Calcula o quanto propostas isncritas em conjunto entre dois ou mais deputados foram aceitas na câmara
        '''
        edges = generateEdges(collab_list)
        for edge in edges:
            if edge in graph:
                graph[edge] += proposal_pertinence
            else:
                graph[edge] = proposal_pertinence
        return graph

    def setCollaborationsSuccess(self):
        '''
        Define atributo de pertiência para cada aresta. Esse atributo representa os projetos entre dois deputados
        que em algum nível foram aceitos pela c6amara
        '''
        for proposal_id in list(self.proposals.keys()):
            if proposal_id in list(self.proposal_authors.keys()):
                proposal_authors = self.proposal_authors[proposal_id]
                proposal_type = self.proposals[proposal_id]['siglaTipo']
                proposal_status = int(self.proposals[proposal_id]['ultimoStatus_idSituacao'])
                n_proposal_weight = proposal_weight[proposal_type]

                status_pertinence = 0
                for status in positive_proposal_status:
                    if(int(proposal_status) == status['status_code']):
                        status_pertinence = status['positive_pertinence']

                pertinence_weighted = n_proposal_weight * status_pertinence
                # proposals with only one author add node weight and will not be considered agai
                if(len(proposal_authors) > 1):
                    self.collab_pertinence = self.addCollabPertinence(
                        self.collab_pertinence, proposal_authors, pertinence_weighted
                    )

    def setDeputiesIndividualProposals(self):
        '''
        Propostas individuais escritas por um deputado e ponderadas por peso de acordo com seu tipo
        '''
        deputies_weight = {}
        for proposal_id in list(self.proposal_authors.keys()):
            authors_list = self.proposal_authors[proposal_id]
            n_authors = len(authors_list)
            # checks if the proposal have only one author
            if (n_authors == 1 and (proposal_id in list(self.proposals.keys()))):
                author_id = authors_list[0]
                proposal_type = self.proposals[proposal_id]['siglaTipo']
                n_proposal_weight = proposal_weight[proposal_type]
                if(author_id in deputies_weight):
                    deputies_weight[author_id] += n_proposal_weight
                else:
                    deputies_weight[author_id] = n_proposal_weight
        self.deputies_proposals = deputies_weight

    def setDeputiesIndividualSuccessProposals(self):
        '''
        Propostas escritas por apenas um deputado que tiveram algum grau de aceitação na camara, ponderadas também
        por peso de acordo com o seu tipo
        '''
        deputies_pertincence = {}
        for proposal_id in list(self.proposals.keys()):
            authors_list = self.proposal_authors[proposal_id]
            n_authors = len(authors_list)
            # checks if the proposal have only one author
            if (n_authors == 1 and (proposal_id in list(self.proposals.keys()))):
                author_id = authors_list[0]
                proposal_type = self.proposals[proposal_id]['siglaTipo']
                proposal_status = int(self.proposals[proposal_id]['ultimoStatus_idSituacao'])
                proposal_weight = proposal_weight[proposal_type]
                status_pertinence = 0

                for status in positive_proposal_status:
                    if(int(proposal_status) == status['status_code']):
                        status_pertinence = status['positive_pertinence']
                pertinence_weighted = proposal_weight * status_pertinence

                if(author_id in deputies_pertincence):
                    deputies_pertincence[author_id] += pertinence_weighted
                else:
                    deputies_pertincence[author_id] = pertinence_weighted

        self.deputies_proposals_pertinence = deputies_pertincence

    def setDeputiesRoleInfluence(self):
        '''
        Calcula a influência de um deputado de acordo com os cargos que este ocupo na câmara nos anos
        referentes as legislaturas selecionadas
        '''
        deputies_weight = {}
        role_weight_dict = role_weights
        for deputy_id, role in self.legislative_roles.iterrows():
            role_name = role['role_name']
            deputy_id = str(deputy_id)
            if (deputy_id in self.deputies_ids):
                if(role_name in role_weight_dict.keys()):
                    weight = role_weight_dict[role_name]
                elif("Líder" in role_name):
                    party_id = role['role_place_id']
                    party_weight = (self.parties[party_id]["members_number"]/513) * 30
                    weight = role_weight_dict["Líder de partido"] * party_weight
                if(deputy_id in deputies_weight):
                    deputies_weight[deputy_id] += weight
                else:
                    deputies_weight[deputy_id] = weight

        self.roles_relevance = deputies_weight

    def setDeputiesRegion(self):
        '''
        Com base na unidade da federação do deputado, adiciona a região a qual ele pertence
        '''
        for deputy_id in self.deputies_ids:
            uf = self.deputies[deputy_id]['uf']
            self.deputies[deputy_id]['region'] = getUfRegion(uf)