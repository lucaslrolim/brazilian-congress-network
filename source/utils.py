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

def plotDict(dict, labels=None, bar_color='#7394CB', y_extra_ticks=None, x_grid=False, y_grid=False, save=False, fill_gap=False, use_x_ticks=True):
    fig = plt.gcf()
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams['axes.titlepad'] = 20

    if (fill_gap):
        barlist = plt.bar(range(len(dict)), list(dict.values()), align='center', color=bar_color, width=1.0)
    else:
        barlist = plt.bar(range(len(dict)), list(dict.values()), align='center', color=bar_color)

    if(use_x_ticks):
        plt.xticks(range(len(dict)), list(dict.keys()), rotation=90)

    if(y_extra_ticks is not None):
        plt.yticks(list(plt.yticks()[0]) + y_extra_ticks)

    if(x_grid and y_grid):
        plt.grid(True, ls='dashed')
    elif(x_grid):
        plt.grid(axis='x', ls='dashed')
    elif(y_grid):
        plt.grid(axis='y', ls='dashed')

    if(labels is not None):
        plt.title(labels['title'])
        plt.xlabel(labels['xlabel'])
        plt.ylabel(labels['ylabel'])
    if(save):
        plt.draw()
        fig.savefig('./images/' + labels['title'] + ".png", dpi=100, bbox_inches="tight")
    plt.show()

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