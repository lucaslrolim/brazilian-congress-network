import datetime
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculateAge(b_date):
    '''
    Calcula a idade baseado na data de nascimento
    '''
    born = datetime.datetime.strptime(b_date, "%Y-%m-%d")
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def getAgeRange(age):
    '''
    Define intervalos de idade para categorizar deputados
    '''
    if(age <= 30):
        return 1
    elif(age > 30 and age <= 50):
        return 2
    elif(age > 50 and age <= 65):
        return 3
    else:
        return 4

def generateEdges(collab_list):
    '''
    Com base em um conjuntos de vértices, gera as combinações de arestas possíveis
    '''
    edge_list = [(collab_list[i],collab_list[j]) for i in range(len(collab_list)) for j in range(i+1, len(collab_list))]
    result = edge_list.copy()
    for element in edge_list:
        result.append((element[1],element[0]))
    return result

def reject_outliers(data_dict, m=2):
    '''
    Dado um dicionário, remove entradas que estejam fora de um determinado número
    de desvios padrão
    '''
    mean = np.mean(list(data_dict.values()))
    std = np.std(list(data_dict.values()))
    result = {}
    for key, val in data_dict.items():
        if(abs(val - mean) < m * std):
            result[key] = val
        else:
            print("excluido valor: ", val)
    return result

def saveTable(self, column_list, column_names, file_name):
    dic = {}
    for i in range(len(column_list)):
        dic[column_names[i]] = column_list[i]
    df = pd.DataFrame.from_dict(dic)
    df.to_csv('./tables/' + file_name + '.csv')

def generateNodePairs(edges):
    '''
    Dada uma lista de arestas, gera todas as combinações possíveis
    '''
    pairs = [(edges[i], edges[j]) for i in range(len(edges)) for j in range(i + 1, len(edges))]
    return pairs

def getThresholdCounts(data, threshold, tolerance=0):
    ret = {"minor": 0, "equal": 0, "bigger": 0}
    for element in list(data.values()):
        if((threshold - tolerance) < element < (threshold + tolerance)):
            ret["equal"] += 1
        elif(element > threshold):
            ret["bigger"] += 1
        elif(element < threshold):
            ret["minor"] += 1
    return ret

def getUfRegion(uf):
    regions_dict = {
        "AM": "Norte",
        "RR": "Norte",
        "AP": "Norte",
        "PA": "Norte",
        "TO": "Norte",
        "RO": "Norte",
        "AC": "Norte",
        "MA": "Nordeste",
        "PI": "Nordeste",
        "CE": "Nordeste",
        "RN": "Nordeste",
        "PE": "Nordeste",
        "PB": "Nordeste",
        "SE": "Nordeste",
        "AL": "Nordeste",
        "BA": "Nordeste",
        "MT": "Centro-Oeste",
        "MS": "Centro-Oeste",
        "GO": "Centro-Oeste",
        "DF": "Distrito Federal",
        "SP": "Sudeste",
        "RJ": "Sudeste",
        "ES": "Sudeste",
        "MG": "Sudeste",
        "PR": "Sul",
        "RS": "Sul",
        "SC": "Sul"
    }
    return regions_dict[uf]