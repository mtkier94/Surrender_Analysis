import os

# path variables
cwd = os.path.dirname(os.path.realpath(__file__))
path_portfolio = os.path.join(cwd, r'DATA')
path_tables = os.path.join(cwd, r'Tables')
path_plots = os.path.join(cwd, r'Plots')

def getDataPath(profile):
    return os.path.join(path_portfolio, r'profile_{}'.format(profile))