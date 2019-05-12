
from MinerFactory import MinerFactory

miners = ["DeputiesMiner", "PartiesMiner", "ProposalsMiner", "RolesMiner", "TSEMiner", "AuthorsMiner"]

mf = MinerFactory(miners, [2014, 2015, 2016, 2017, 2018], [55], start_date="2015-01-01", end_date="-2019-01-01")
mf.buildAll()