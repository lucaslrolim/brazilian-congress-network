import click
import ast
from miners import MinerFactory
from source import NetworkBuilder
import os

def dateStringInterval(dates):
    min_year = min(dates)
    max_year = max(dates)
    min_string = "{}-01-01".format(min_year)
    max_string = "{}-01-01".format(max_year)
    return min_string, max_string

@click.command()
@click.option('--extract_data', nargs=3, default=False, help="""
Extrai os dados desejados no intervalo de tempo fornecido. \n\n
Recebe como argumento uma lista de miners, uma lista de anos e uma lista de legislaturas. \n\n
[miners] [anos] [legislaturas] \n
Os miners disponíveis são: \n
APIProposalMiner -> proposições  usando API, \n
AuthorsMiner -> autores das proposições, \n
DeputiesMiner -> deputados ativos , \n
PartiesMiner -> partidos representados, \n
ProposalsMiner -> proposições apresentadas, \n
RolesMiner -> cargos dos deputados, \n
TSEMiner -> características pessoais \n
.""")
@click.option('--build_network', type=click.Choice(['weighted', 'not_weighted']), help='Constrói a rede de coautoria de projetos. Pode ou não considerar arestas com peso.')


def exec_task(extract_data, build_network):
    if(extract_data):
        os.chdir('./miners') 
        miners = ast.literal_eval(extract_data[0])
        years = ast.literal_eval(extract_data[1])
        legislatures = ast.literal_eval(extract_data[2])
        start_date, end_date = dateStringInterval(years)

        mf = MinerFactory.MinerFactory(miners, years, legislatures, start_date, end_date)
        mf.buildAll()

    if(build_network):
        os.chdir('./source') 
        nb = NetworkBuilder.NetworkBuilder()
        if(build_network == 'weighted'):
            nb.buildNetwork(True)
        else:
            nb.buildNetwork(False)
        nb.saveNetWork()

if __name__ == '__main__':
    exec_task()

