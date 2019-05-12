role_weights = {
    "Presidente": 10,
    "1º Vice-Presidente": 7,
    "1º Secretário":5,
    "2º Secretário":4,
    "3º Secretário":4,
    "4º Secretário":4,
    "1º Suplente de Secretário": 2,
    "2º Suplente de Secretário": 2,
    "3º Suplente de Secretário": 2,
    "4º Suplente de Secretário": 2,
    "Líder de partido": 5,
    "Titular": 0.5,
    "Suplente": 0.25
}

proposal_weight = {
    "PEC": 1,
    "PL": 1,
    "PLN": 1,
    "PLP": 1,
    "PLV": 1,
    "PLC": 1
}

positive_proposal_status = [
    {
        "status_name": "Aguardando Redação Final",
        "positive_pertinence": 1,
        "status_code": 929
    },
    {
        "status_name": "Enviada ao Arquivo",
        "positive_pertinence": 1,
        "status_code": 930
    },
    {
        "status_name": "Aguardando Remessa ao Arquivo",
        "positive_pertinence": 1,
        "status_code": 931
    },
    {
        "status_name": "Aguardando Recebimento para Publicação - Relatadas",
        "positive_pertinence": 1,
        "status_code": 1000
    },
    {
        "status_name": "Aguardando Encaminhamento à Publicação",
        "positive_pertinence": 1,
        "status_code": 1020
    },
    {
        "status_name": "Transformado em Norma Jurídica",
        "positive_pertinence": 1,
        "status_code": 1140
    },
    {
        "status_name": "Aguardando Sanção",
        "positive_pertinence": 1,
        "status_code": 1150
    },
    {
        "status_name": "Aguardando Remessa à Sanção",
        "positive_pertinence": 1,
        "status_code": 1160
    },
    {
        "status_name": "Aguardando Envio à Redação Final",
        "positive_pertinence": 1,
        "status_code": 1270
    },
    {
        "status_name": "Aguardando Promulgação",
        "positive_pertinence": 1,
        "status_code": 1294
    }
]

node_parameters = {
    "role": 1,
    "proposal": 1
}