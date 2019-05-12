def addProposalType(string, types):
    result = string
    for p_type in types:
        result = result + "&siglaTipo={}".format(p_type)
    return result


def addProposalYear(string, types):
    result = string
    for p_year in types:
        result = result + "&ano={}".format(p_year)
    return result


def addProposalSituation(string, types):
    result = string
    for p_situation in types:
        result = result + "&idSituacao={}".format(p_situation)
    return result


def addLegislature(string, types):
    result = string
    for p_legislature_id in types:
        result = result + "&idLegislatura={}".format(p_legislature_id)
    return result


def addStatus(string, status):
    if(status):
        return string + "&idLegislatura={}".format("true")
    else:
        return string + "&idLegislatura={}".format("false")


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total: 
        print()