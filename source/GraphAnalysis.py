import random
import numpy as np
import networkx as nx
import math
import collections
import matplotlib.pyplot as plt
import copy
import pandas as pd
from utils import reject_outliers
from utils import generateNodePairs
from utils import getThresholdCounts


class GraphAnalysis:
    '''
    Realiza análises da rede de coautorias de proposição de lei, extraindo métricas como
    modularidade, estatísticas de grau e arestas, informações dos nós e assortatividade.

    :param G: rede no formato da biblioteca NetworkX
    '''
    def __init__(self, G):
        self.graph = G
        self.nodes_number = self.graph.number_of_nodes()
        self.edges_number = self.graph.number_of_edges()
        self.degree_info = self.setDegreeInfo()
        self.components_number = self.setNumberOfComponents()
        self.global_clustering_index = nx.average_clustering(self.graph)
        self.everage_local_clustering_index = None
        self.average_shortest_path = None
        self.pseudo_diameter = None
        self.fillNullAttribute('party')

    def setDeputiesList(self, deputies_dict):
        self.deputies_list = list(deputies_dict.keys())

    def getGraph(self):
        return self.graph

    def getAverageShortestPath(self):
        return self.average_shortest_path

    def getDiameter(self):
        return self.pseudo_diameter

    def getGlobalClustering(self):
        return self.global_clustering_index

    def getNumberOfNodes(self):
        return self.nodes_number
  
    def getNumberOfEdges(self):
        return self.edges_number

    def setDegreeInfo(self):
        '''
        Cria dicionário com informações sobre grau mínimo, médio e máximo da rede e seus respectivos vértices
        '''
        g_max = {"node": None, "value": 0}
        g_min = {"node": None, "value": math.inf}
        zeros = 0
        g_sum = 0
        for node_degree in list(self.graph.degree().items()):

            if(node_degree[1] > g_max['value']):
                g_max['node'] = node_degree[0] 
                g_max['value'] = node_degree[1]

            if(node_degree[1] < g_min['value']):
                g_min['node'] = node_degree[0] 
                g_min['value'] = node_degree[1] 

            if(node_degree[1] == 0):
                zeros += 1

            g_sum += node_degree[1]
        result = {"min": g_min, "max": g_max, "mean": g_sum/len(self.graph.degree()), "zeros": zeros}
        return result

    def getDegreeInfo(self):
        return self.degree_info

    def getDensity(self):
        return nx.density(self.graph)

    def setNumberOfComponents(self):
        return nx.number_connected_components(self.graph)

    def getNumberOfComponents(self):
        return self.components_number

    def getRelativeSizeLargeComponent(self):
        largest_cc = max(nx.connected_components(self.graph), key=len)
        largest_cc_size = len(largest_cc)
        return largest_cc_size/self.getNumberOfNodes()

    def getDeputyDegree(self, deputy_id):
        print("O grau do deputado {} é: {}".format(deputy_id, self.graph.degree()[str(deputy_id)]))

    def getDeputyInfo(self, deputy_id):
        '''
        Retorna informações do parlamentar vinculado a um determinado id que identifica um vértice da rede
        '''
        nodes = self.graph.nodes(data=True)
        for node in nodes:
            if(node[0] == deputy_id):
                return node

    def fillNullAttribute(self, attribute_id):
        '''
        Preeenche atributos vazios (não preenchidos pela câmara) com o valor null para um determinado atributo
        '''
        for node in self.graph.nodes(data=True):
            try:
                node[1][attribute_id] is True
            except:
                node[1][attribute_id] = 'null'

    def printSummary(self):
        '''
        Exibe na tela um resumo das principais propriedades da rede
        '''
        degree_info = self.getDegreeInfo()
        largest_cc = max(nx.connected_components(self.graph), key=len)
        largest_cc_size = len(largest_cc)
        column_list = [
            self.getNumberOfNodes(),
            self.getNumberOfEdges(),
            nx.density(self.graph),
            self.getNumberOfComponents(),
            nx.average_clustering(self.graph),
            largest_cc_size,
            self.getRelativeSizeLargeComponent(),
            degree_info['min']['value'],
            degree_info['max']['value'],
            degree_info['mean'],
            degree_info['zeros']
        ]

        column_names = [
            'Número de vértices',
            'Número de arestas',
            'Densidade',
            'Número de componentes conexas',
            'Coeficiente de clusterização',
            'Tamanho da maior componente conexa',
            'Tamanho relativo da maior componente conexa',
            'Maior grau',
            'menor grau',
            'Grau médio',
            'Vértices com grau zero' 
        ]
        for i in range(len(column_list)):
            print(column_names[i] + ' : {}'.format(column_list[i]))

    def getNodeAttributeNames(self, ignore_att=['label', 'style', 'weight', 'age', 'education']):
        '''
        Retorna uma lista com os nomes dos atributos que um vértice possui
        '''
        sample_node = self.getGraph().nodes(data=True)[0]
        attribute_names = list(sample_node[1].keys())
        attribute_names = list(set(attribute_names) - set(ignore_att))
        return attribute_names

    def assortativitySummary(self, attributes_to_analyse=None):
        '''
        Retorna um dicionário coma  assortatividade para cada atributo da rede
        '''
        if(attributes_to_analyse is None):
            attributes_to_analyse = self.getNodeAttributeNames()

        attribute_assortativity = {}
        for attribute in attributes_to_analyse:
            assortativity = nx.attribute_assortativity_coefficient(self.graph, attribute)
            attribute_assortativity[attribute] = assortativity

        return attribute_assortativity

    def getDegreeAssorativity(self):
        return nx.degree_assortativity_coefficient(self.graph)

    def getPageRank(self, alpha=0.9):
        return nx.pagerank(self.graph, alpha=alpha)

    def mixingDictingToProb(self, mixing_dict):
        '''
        Normaliza matriz de mixagem para valores de probabilidade
        '''
        prob_dict = copy.deepcopy(mixing_dict)
        for attribute in list(prob_dict.keys()):
            for sub_attribute in mixing_dict[attribute].keys():
                prob_dict[attribute][sub_attribute] = mixing_dict[attribute][sub_attribute] / np.sum(list(mixing_dict[attribute].values()))
        return prob_dict

    def getJaccardSimilarity(self):
        '''
        Coeficiente de Jaccard da rede
        '''
        return nx.jaccard_coefficient(self.graph)

    def getJaccardSimilarityPair(self, node_pair):
        '''
        Similaridade de Jaccard entre dois vértices
        '''
        return nx.jaccard_coefficient(self.graph, [node_pair])

    def getNodeAttribute(self, node_id, attribute):
        '''
        Valor que um determinado atributo assume em um dado vértice
        '''
        return dict(self.graph.nodes(data='true'))[node_id][attribute]

    def getNodeHeterogeneity(self, node_id, het_attributes=['uf', 'party'], norm=False):
        het = self.getHeterogeneity(het_attributes, norm)
        return het[node_id]

    def getNodesByAttribute(self, attribute_id):
        '''
        Retorna um dicionário em que as são cada valor que o atributo fornecido pode assumir na rede.
        Cada uma dessas chaves tem associada a si uma lista com N vértices da rede
        '''
        attribute_nodes = {}
        attribute_nodes['null'] = []
        attribute = attribute_id
        for node in self.graph.nodes(data=True):
            try:
                attribute_name = node[1][attribute]
                node_id = node[0]
                if(attribute_name in attribute_nodes):
                    attribute_nodes[attribute_name].append(node_id)
                else:
                    attribute_nodes[attribute_name] = []
            except:
                attribute_nodes['null'].append(node[0])

        return attribute_nodes

    def countNodesByAttribute(self):
        '''
        Retorna um dicionário de dicionários, com o número de vértices que possui cada um dos subgrupos possíveis
        de cada atributo.

        Por exemplo, nesse dicionário temos a chave ¨party¨, que possui como valor um dicionário em que cada chave é
        um partido político e o valor é o número de deputados com aquele partido na rede.
        '''
        count_attributes = {}
        attributes_list = ['age_range', 'education', 'party', 'sex', 'uf', 'ethnicity', 'education_tse']

        for att in attributes_list:
            count_attributes[att] = {}

        for node in self.graph.nodes(data=True):
            try:
                    for node_att in list(node[1].items()):
                        node_att_name = node_att[0]
                        node_att_value = node_att[1]
                        if node_att_name in attributes_list:
                            if(node_att_value in count_attributes[node_att_name]):
                                count_attributes[node_att_name][node_att_value] += 1
                            else:
                                count_attributes[node_att_name][node_att_value] = 1
            except:
                pass

        return count_attributes

    def adjMatrix(self, weighted=False):
        size = len(self.graph.nodes())
        adj_matrix = np.zeros((size, size))
        matrix_map = {}
        matrix_map['index_to_dep'] = {}
        count = 0

        for deputy_id in self.graph.nodes():
            matrix_map[deputy_id] = count
            matrix_map['index_to_dep'][count] = deputy_id
            count += 1
        for edge in self.graph.edges(data=True):
            deputy_1 = edge[0]
            deputy_2 = edge[1]
            if(weighted):
                weight = edge[2]['weight']
            else:
                weight = 1

            adj_matrix[matrix_map[deputy_1]][matrix_map[deputy_2]] += weight
            adj_matrix[matrix_map[deputy_2]][matrix_map[deputy_1]] += weight
        return adj_matrix, matrix_map

    def joinEdgesAttFraction(self, attribute, adj_matrix, m_map):
        # total_weight retorma 2m, porem nao vamos iterar sobre as arestas duas vezes
        total_weight = adj_matrix.sum() / 2
        result = {}
        for edge in self.graph.edges():
            d1 = edge[0]
            d2 = edge[1]
            deputy_1 = self.getDeputyInfo(d1)
            deputy_2 = self.getDeputyInfo(d2)
            att_1 = deputy_1[1][attribute]
            att_2 = deputy_2[1][attribute]

            if(att_1 not in result):
                result[att_1] = 0
            if(att_1 == att_2):
                result[att_1] += adj_matrix[m_map[d1]][m_map[d2]]

        normalized_result = {k: v / total_weight for k, v in result.items()}
        return normalized_result

    def incidenceEdgesAttFraction(self, attribute, weighted=False):
        result = {}
        weights = self.getSumEdgeWeights()
        total_weight = 0
        for node in self.graph.nodes(data=True):
            deputy_id = node[0]
            deputy_att = node[1][attribute]

            if(weighted):
                weight = weights[deputy_id]
            else:
                weight = len(self.graph.neighbors(deputy_id))

            total_weight += weight

            if(deputy_att in result):
                result[deputy_att] += weight
            else:
                result[deputy_att] = weight

        normalized_result = {k: v / total_weight for k, v in result.items()}
        return normalized_result

    def modularity(self, attribute, weighted=False):
        """
        Retorna a modulatidade geral do atributo baseada na equação proposta por Newman
        e também um dicionário com as modularidades de todos os valores possíveis do domínio
        do atributo selecionado
        """
        adj_matrix, m_map = self.adjMatrix(weighted)
        modularity_result = {}
        e_r = self.joinEdgesAttFraction(attribute, adj_matrix, m_map)
        a_r = self.incidenceEdgesAttFraction(attribute, weighted)

        att_possibilies = list(e_r.keys())
        att_modularity = 0
        for att in att_possibilies:
            modularity_result[att] = e_r[att] - pow(a_r[att], 2)
            att_modularity += modularity_result[att]

        return att_modularity, modularity_result

    def modularitySummary(self, attributes_to_analyse=None, weighted=False):
        if(attributes_to_analyse is None):
            attributes_to_analyse = self.getNodeAttributeNames()

        attribute_modularity = {}
        for attribute in attributes_to_analyse:
            modularity = self.modularity(attribute, weighted)
            attribute_modularity[attribute] = modularity

        return attribute_modularity

    def getExpectedWeightByAttribute(self, homophily_attribute, norm=True):
        """
        Calcula o valor esperado da distribuição de pesos de arestas incidentes por atributo do vértice
        """
        attribute_edges_weight = {}
        graph_edge_weight = 0
        for edge in self.graph.edges(data=True):
            node_att_1 = self.getNodeAttribute(edge[0], homophily_attribute)
            node_att_2 = self.getNodeAttribute(edge[1], homophily_attribute)
            edge_weight = edge[2]['weight']

            graph_edge_weight += edge_weight

            if(node_att_1 in attribute_edges_weight):
                attribute_edges_weight[node_att_1] += edge_weight
            else:
                attribute_edges_weight[node_att_1] = edge_weight

            if(node_att_2 in attribute_edges_weight):
                attribute_edges_weight[node_att_2] += edge_weight
            else:
                attribute_edges_weight[node_att_2] = edge_weight

        if(norm):
            attribute_edges_weight = {k: (v/2) / graph_edge_weight for k, v in attribute_edges_weight.items()}

        return attribute_edges_weight

    def getSumEdgeWeights(self):
        """
        Retorna dicionário com a soma dos pesos das arestas incidentes a cada vértice
        """
        weights = {}
        for node in self.graph.nodes(data=True):
            node_id = node[0]
            node_edge_weights = 0
            for neighbor in self.graph.neighbors(node_id):
                node_edge_weights += self.graph[node_id][neighbor]['weight']
            weights[node_id] = node_edge_weights
        return weights

    def nodesModularityByAttribute(self, attribute, weighted=False):
        """
        Retorna modularidade individual de todos os nós para um determinado atributo
        """
        network_result = {}
        nodes_att = {}
        for node in self.graph.nodes(data=True):
            node_id = node[0]
            base_attribute_value = node[1][attribute]
            nodes_att[node_id] = base_attribute_value
            w_homo = 0
            w_total = 0
            for neighbor_id in self.graph.neighbors(node_id):
                edge_weigth = self.graph[node_id][neighbor_id]['weight']
                neighbor = self.graph.node[neighbor_id]

                if(weighted is True):
                    weight = edge_weigth
                else:
                    weight = 1

                w_total += weight
                if(attribute in neighbor):                  
                    if(neighbor[attribute] == base_attribute_value):
                        w_homo += weight

            if(w_total > 0):
                network_result[node_id] = w_homo/w_total
            else:
                network_result[node_id] = None

        result = {}
        expected_distribution = self.getExpectedWeightByAttribute(attribute)

        for node_id, homophily_percent in network_result.items():
            if(homophily_percent is not None):
                result[node_id] = homophily_percent - expected_distribution[nodes_att[node_id]]
            else:
                result[node_id] = None

        return result

    def nodesModularity(self, use_weight=False, attributes_to_analyse=None):
        """
        Modularidade individual do vértice, que é calculada com base no percentual do peso de arestas
        incidentes ao vértice que são com vizinhos homofílicos
        """
        if(attributes_to_analyse is not None):
            attributes_to_analyse = self.getNodeAttributeNames()
        nodes_modularity = {}
        node_degrees = self.graph.degree()
        
        for node_id, degree in node_degrees.items():
            if(degree != 0):
                nodes_modularity[node_id] = 0

        for attribute in attributes_to_analyse:
            modularity = self.nodesModularityByAttribute(attribute, weighted=use_weight)
            for node_id in list(nodes_modularity.keys()):
                nodes_modularity[node_id] += modularity[node_id]

        return nodes_modularity

    def plotDegreeDistribution(self, color='#7394CB'):
        data = nx.degree_histogram(self.graph)
        labels = {"title": 'CCDF grau dos vértices', 'xlabel': 'número de coautores', 'ylabel': '% de deputados'}
        self.plotDistributionCCDF(data, labels, color)

    def plotWeightDistribution(self, color='#7394CB'):
        weights = list(self.getSumEdgeWeights().values())
        data = np.zeros(int(max(weights)))
        for i in range(len(data)):
            i_count = weights.count(i)
            data[i] = i_count
        labels = {"title": 'CCDF do peso de arestas incidentes', 'xlabel': 'peso incidente', 'ylabel': '% de deputados'}
        self.plotDistributionCCDF(data, labels, color)

    def plotDistributionCCDF(self, data, labels, color='#7394CB'):
        cumulative = np.cumsum(data)
        ccdf = []
        for element in cumulative:
            prob = (element - cumulative[0])/(max(cumulative)-min(cumulative))
            ccdf.append(1 - prob)    
        label = list(range(len(data)))
        plt.plot(label, ccdf, color=color)
        plt.title(labels['title'])
        plt.xlabel(labels['xlabel'])
        plt.ylabel(labels['ylabel'])
        plt.show()

    def plotMixPattern(self, attribute, attribute_value):
        mix_dict = nx.attribute_mixing_dict(self.graph, attribute)
        norm_mix = self.mixingDictingToProb(mix_dict)

        # O atributo abaixo define o valor de atributo que desejamos analisar o padrão de mixagem
        # por exemplo, mixagem para o partido PT ou para o sexo Masculino
        D = norm_mix[attribute_value]

        plt.bar(range(len(D)), list(D.values()), align='center')
        plt.xticks(range(len(D)), list(D.keys()), rotation='vertical')

        plt.show()

    def plotDict(self, dict, labels=None, bar_color='#7394CB', y_extra_ticks=None, x_grid=False, y_grid=False, save=False, fill_gap=False, use_x_ticks=True):
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

################################################################
### Métodos legados, não utilizados na versão final do paper ###
################################################################

    def mixingMatrix(self, attribute, weighted=False, norm=True):
        mix_dict = {}
        matrix_dict = {}
        # itera nas arestas da rede
        for edge in self.graph.edges(data=True):
            # coleta informacoes dos deputados na ponta das arestas
            deputy_1 = self.getDeputyInfo(edge[0])
            deputy_2 = self.getDeputyInfo(edge[1])
            # verifica o valor dos deputados para o atributo de homofilia selecionado
            att_1 = deputy_1[1][attribute]
            att_2 = deputy_2[1][attribute]

            # peso da aresta, que podera ou nao ser considerado
            weight = edge[2]['weight']

            # monta dinamicamente a matriz de adjacencia em um dicionario
            if(att_1 not in mix_dict):
                mix_dict[att_1] = max(mix_dict.values(), default=-1) + 1
            if(att_2 not in mix_dict):
                mix_dict[att_2] = max(mix_dict.values(), default=-1) + 1

            tuple_1 = (mix_dict[att_1], mix_dict[att_2])
            tuple_2 = (mix_dict[att_2], mix_dict[att_1])

            if(weighted):
                relationship_force = weight
            else:
                relationship_force = 1

            if(tuple_1 in matrix_dict):
                matrix_dict[tuple_1] += relationship_force
            else:
                matrix_dict[tuple_1] = relationship_force

            if(tuple_2 in matrix_dict):
                matrix_dict[tuple_2] += relationship_force
            else:
                matrix_dict[tuple_2] = relationship_force
        
        # monta a matrix no formato de arrays e normaliza os valores
        matrix_size = len(mix_dict.keys())
        matrix = np.zeros((matrix_size, matrix_size))

        for i in range(matrix_size):
            for j in range(matrix_size):
                if (i, j) in matrix_dict:
                    matrix[i][j] = matrix_dict[(i,j)]
                else:
                    matrix[i][j] = 0

        # converte valores totais para porcentagem
        if(norm):
            for i in range(matrix_size):
                row_sum = np.sum(matrix[i])
                for j in range(matrix_size):
                    matrix[i][j] = matrix[i][j]/row_sum
        
        return matrix, mix_dict

    def getHeterogeneity(self, het_attributes=['uf', 'party'], norm=False):
        heterogeneity = {}
        for att in het_attributes:
            w_jaccard = self.getWeightedJaccard(att, norm)
            expected_homophily = self.getExpectedHomophily(att, norm)
            for deputy_id in list(w_jaccard.keys()):
                if(deputy_id in heterogeneity):
                    heterogeneity[deputy_id] += (
                        w_jaccard[deputy_id] - expected_homophily[deputy_id])/max(1, expected_homophily[deputy_id])
                else:
                    heterogeneity[deputy_id] = (
                        w_jaccard[deputy_id] - expected_homophily[deputy_id])/max(1, expected_homophily[deputy_id])       
        return heterogeneity

    def getExpectedHomophily(self, homophily_attribute, norm=False):
        '''
        Calcula o valor esperado de homofilia para um vétice, dados seus atributos
        Pensar em como implementar essse método
        '''

        nodes_edge_weights = self.getSumEdgeWeights()
        # valor esperado da soma dos pesos para vértices com o atributo de homofilia fornecido como argumento
        expected_weight = {}
        expected_homophily = {}
        expected_degree = self.countNodesByAttribute()[homophily_attribute]
        expected_degree = {k: v / self.getNumberOfNodes() for k, v in expected_degree.items()}
        expected_edge_weights = self.getExpectedWeightByAttribute(homophily_attribute)
        for node in self.graph.nodes(data=True):
            node_id = node[0]
            attribute_name = node[1][homophily_attribute]
            sum_edges_weights = nodes_edge_weights[node_id]
            node_degree = max(self.graph.degree()[str(node_id)], 1)
            if(node_id in expected_weight):
                expected_weight[node_id] = sum_edges_weights * expected_edge_weights[attribute_name]
                w_mean = sum_edges_weights/node_degree
                # valor esperado do peso dividido pelo valor esperado do numero de vizinhos com aquele atributo
                w_mean_homophily = expected_weight[node_id] / (max(1, node_degree * expected_degree[node_id]))
                expected_homophily[node_id] = (w_mean_homophily - w_mean) / max(w_mean, 1)
            else:
                # vértices de grau zero
                expected_homophily[node_id] = 0
        return expected_homophily

    def jaccardByAttribute(self, attribute_id):
        '''
        Coeficiente de Jaccard condicionado a algum dos atibutos da rede
        '''
        nodes_attr = self.getNodesByAttribute(attribute_id)
        jaccard_dict = {}
        for party in list(nodes_attr.keys()):
            jaccard_values = []
            preds = nx.jaccard_coefficient(self.graph, generateNodePairs(nodes_attr[party]))
            jaccard_values = []
            for u, v, p in preds:
                jaccard_values.append(p)
            if(len(jaccard_values) > 1):
                jaccard_dict[party] = np.mean(jaccard_values)
            else:
                print("Too low data to compute Jaccard on: ", party)
        return jaccard_dict

    def getWeightedJaccard(self, homophily_attribute, norm=False):
        '''
        Recebe como argumento o atributo o qual se deseja analisar a homofilia e retorna como resultado
        um dicionário com a métrica proposta no trabalho e baseada no índice de Jaccard.
        O cálculo do índice da métrica de homofilia com pesos é feito basicamente comparando
        o peso das arestas do vértice que são para outros vértices com o mesmo atributo
        de homofilia (argumento função) e a soma do peso das arestas desse
        vértice em geral.
        '''
        w_jaccard = {}
        count_att = self.countNodesByAttribute()[homophily_attribute]
        sum_count_att = sum(list(count_att.values()))
        for node in self.graph.nodes(data=True):
            try:
                attribute_name = node[1][homophily_attribute]
                node_id = node[0]
                general_weigths = []
                homophily_weigths = []

                for neighbor in self.graph.neighbors(node_id):
                    edge_weigth = self.graph[node_id][neighbor]['weight']
                    general_weigths.append(edge_weigth)
                    if(homophily_attribute in self.graph.node[neighbor]):                  
                        if(self.graph.node[neighbor][homophily_attribute] == attribute_name):
                            homophily_weigths.append(edge_weigth)
                    else:
                        print("Vértice {} não possui o atributo {}".format(neighbor, homophily_attribute))

                homophily_mean = float(sum(homophily_weigths))/max(len(homophily_weigths), 1)

                general_mean = float(sum(general_weigths))/max(len(general_weigths), 1)
                # normalization constant to remove bias
                norm_beta = 1
                if (norm):
                    alpha = (1/len(count_att)) + (1 - (count_att[attribute_name]/sum_count_att)) * norm_beta
                else:
                    alpha = 1

                sub = (homophily_mean * (alpha) - general_mean)
                w_jaccard[node_id] = (sub)/max(general_mean, 1)

            except:
                w_jaccard[node_id] = None
        w_jaccard = {key: val for key, val in w_jaccard.items() if (val is not None)}

        return w_jaccard

    def getWeightedJaccardByAttribute(self, homophily_attribute, conditional_attribute, norm=False, modulation=None):
        '''
        Recebe como argumento o atributo ao qual se deseja analisar a homofilia e um subgrupo dentro desse atributo.
        Por exemplo, ¨party¨ e ¨PT¨.

        Retorna a métrica de homofilia média dentre os vértices pertencentes ao subgrupo informado como argumento.
        '''
        conditional_nodes = self.getNodesByAttribute(homophily_attribute)[conditional_attribute]
        w_jaccard = self.getWeightedJaccard(homophily_attribute, norm)
        sum_w = []

        if(modulation is not None):
            w_jaccard_clean = reject_outliers(w_jaccard, modulation)
        else:
            w_jaccard_clean = w_jaccard

        for node in conditional_nodes:
            if(node in w_jaccard_clean):
                w_j_node = w_jaccard_clean[node]
                sum_w.append(w_j_node)
        return np.mean(sum_w)

    def conditionalWJaccardDict(self, attribute, norm=False, module_param=None):
        '''
        Recebe como argumento o atributo ao qual se deseja analisar a homofilia, como por exemplo ¨party¨.
        
        Retorna um dicionário com a a homofilia média para cada um dos subgrupos existentes para esse atributo.
        Como, por exemplo, a homofilia média e cada um dos partidos políticos.
        '''
        jaccard_dict = {}
        nodes_attr = self.getNodesByAttribute(attribute)

        for conditional_att in list(nodes_attr.keys()):
            jaccard_dict[conditional_att] = self.getWeightedJaccardByAttribute(
                attribute, conditional_att, norm, module_param
                )

        return jaccard_dict