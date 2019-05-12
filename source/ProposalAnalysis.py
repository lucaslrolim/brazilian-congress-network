import numpy as np
import model_parameters
import networkx as nx
import pandas as pd

class ProposalAnalysis:
    def __init__(self, proposal_authors, proposals):
        self.proposal_authors = proposal_authors
        self.proposals = proposals

    def getPrAuthorsInfo(self, graph_analysis_obj, pagerank_alpha=0.9):
        '''
        Soma dos pageranks dos autores de uma proposicao em um dado grafo
        '''
        proposal_authors_info = {}

        pagerank_dict = graph_analysis_obj.getPageRank(pagerank_alpha)
        heterogeneity_dict = graph_analysis_obj.getHeterogeneity()

        for proposal_id in list(self.proposals.keys()):
            if(proposal_id in list(self.proposal_authors.keys())):
                authors_list = self.proposal_authors[proposal_id]
                proposal_authors_info[proposal_id] = {}
                proposal_authors_info[proposal_id]["authors_number"] = len(authors_list)
                p_pagerank = 0
                p_weight = 0
                p_heterogeneity = 0
                for author_id in authors_list:
                    author_id = str(author_id)
                    # desconsidera autores fora da legislatura atual
                    if author_id in pagerank_dict:
                        p_pagerank += pagerank_dict[author_id]
                        p_weight += graph_analysis_obj.getNodeAttribute(author_id, 'weight')
                        p_heterogeneity += heterogeneity_dict[author_id]

                proposal_authors_info[proposal_id]["authors_number"] = len(authors_list)
                proposal_authors_info[proposal_id]["pagerank"] = p_pagerank
                proposal_authors_info[proposal_id]["authors_weight"] = p_weight
                proposal_authors_info[proposal_id]["authors_heterogeneity"] = p_heterogeneity/len(authors_list)

        return proposal_authors_info

    def getPrSituation(self):
        '''
        Dicionário com 1 se a proposição for de alguma forma aprovada e 0 caso contrário
        '''
        proposals_situation = {}
        for proposal_id in list(self.proposals.keys()):
            proposal_status = int(self.proposals[proposal_id]['ultimoStatus_idSituacao'])
            status_pertinence = 0
            for status in model_parameters.positive_proposal_status:
                if(int(proposal_status) == status['status_code']):
                    status_pertinence = status['positive_pertinence']
            proposals_situation[proposal_id] = status_pertinence
        return proposals_situation

    def getPrSituation(self, graph_analysis_obj):
        '''
        Retorna dicionario onde a chave é a proposicao e value é dicionario com pagerank e situacao
        '''
        pagerank_situation_relationship = {}
        p_info = self.getPrAuthorsInfo(graph_analysis_obj)
        p_situation = self.getPrSituation()

        # Algumas proposicaoes estao em uma lista e nao em outra pois em uma das listas
        # filtramos por tipo de propsicao e na outra nao. Bem como retiramos as com dados faltantes
        # no campo de autor
        for proposal_id in list(p_info.keys()):
            if proposal_id in list(p_situation.keys()):
                pagerank_situation_relationship[proposal_id] = {}
                pagerank_situation_relationship[proposal_id]["pagerank"] = p_info[proposal_id]["pagerank"]
                pagerank_situation_relationship[proposal_id]["authors_number"] = p_info[proposal_id]["authors_number"]
                pagerank_situation_relationship[proposal_id]["authors_weight"] = p_info[proposal_id]["authors_weight"]
                pagerank_situation_relationship[proposal_id]["authors_heterogeneity"] = p_info[proposal_id]["authors_heterogeneity"]
                pagerank_situation_relationship[proposal_id]["situation"] = p_situation[proposal_id]
   
        return pagerank_situation_relationship

    def getPrSummary(self):
        total = len(list(self.getPrSituation()))
        approved = sum(list(self.getPrSituation()))
        return {"approved": approved, "rejected": total - approved}
 
    def countPrAttribute(self, attribute):
        attribute_count = {}
        for proposal in list(self.proposals.values()):
            att_value = proposal[attribute]
            if(att_value in attribute_count):
                attribute_count[att_value] += 1
            else:
                attribute_count[att_value] = 1
        return attribute_count

    def convertToPandas(self, data):
        return pd.DataFrame(data).T
