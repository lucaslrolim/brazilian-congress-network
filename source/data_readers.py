import pandas as pd
import numpy as np


def getDeputies():
    df_deputies = pd.read_csv("../data/deputies_info.csv", sep=',')
    df_deputies.set_index('index', inplace=True)
    return df_deputies.to_dict('index')
    

def getInfoTSE():
    tse_info = pd.read_csv("../data/candidates_tse_info.csv", sep=',')
    tse_info['CPF'].fillna(0.0).astype(int)
    tse_info.set_index('CPF', inplace=True)
    return tse_info.to_dict('index')


def getAuthors():
    """
    Retorna um dicionário em que as chaves são as proposições de lei e os valores são os ids
    dos deputados autores dessas proposições.
    """
    df_authors = pd.read_csv("../data/authors_info.csv", sep=',')
    proposal_authors_dict = {}

    for index, row in df_authors.iterrows():
        if(int(row['codTipoAutor']) == 10000):
            proposal_id = int(row['idProposicao'])
            if(row['idProposicao'] in proposal_authors_dict):
                proposal_authors_dict[proposal_id].append(row['idAutor'])
            else:
                proposal_authors_dict[proposal_id] = [row['idAutor']]
    return proposal_authors_dict


def getParties():
    """
    Retorna dicionário de partidos políticos, bem como informações sobre seu número de membros
    """
    df_parties = pd.read_csv("../data/parties_info.csv", sep=',')
    df_parties.set_index('index', inplace=True)
    return df_parties.to_dict('index')


def getRoles():
    """
    Retorna já ocupados por um deputado na câmara.
    """
    df_roles = pd.read_csv("../data/roles_info.csv", sep=',')
    df_roles.set_index('deputy_id', inplace=True)
    df_roles.drop_duplicates()
    return df_roles


def getProposals():
    df_proposals = pd.read_csv("../data/proposals_info.csv", sep=',')
    df_proposals.set_index('id', inplace=True)
    proposals_dict = df_proposals.to_dict('id')
    return {int(k): v for k, v in proposals_dict.items()}