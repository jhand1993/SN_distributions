import numpy as np
from config import *
import os

# import dragan data
datafitres = 'dragan2.fitres'
os.chdir(homedir + '/dragan')
with open(datafitres, 'r') as f:
    datavars = f.readlines()[1]
with open(datafitres, 'r') as f:
    datavarscount = int(f.readlines()[0].split(' ')[1]) - 2
datavars = datavars.split()
datavars.remove('VARNAMES:')
datavars.remove('FIELD')
datavars.remove('CID')
columns = tuple(list(range(2, 5)) + list(range(6, 42)))
dragan = np.loadtxt(datafitres, dtype=float, skiprows=18, usecols=columns)
for i in range(datavarscount):
    if datavars[i] == 'IDSURVEY':
        dataids = np.copy(dragan[:, i])
    elif datavars[i] == 'zCMB':
        datazcmb = np.copy(dragan[:, i])
    elif datavars[i] == 'HOST_LOGMASS':
        datahostmass = np.copy(dragan[:, i])
    elif datavars[i] == 'c':
        datac = np.copy(dragan[:, i])
    elif datavars[i] == 'x1':
        datax1 = np.copy(dragan[:, i])
    elif datavars[i] == 'MURES':
        datamures = np.copy(dragan[:, i])

sdss = np.asarray([x == 1 for x in dataids])
snls = np.asarray([x == 4 for x in dataids])
lowz = np.asarray([x <= 0.1 for x in datazcmb])

sdssmass = np.copy(datahostmass[sdss])
sdssc = np.copy(datac[sdss])
sdssx1 = np.copy(datax1[sdss])
sdssmures = np.copy(datamures[sdss])

snlsmass = np.copy(datahostmass[snls])
snlsc = np.copy(datac[snls])
snlsx1 = np.copy(datax1[snls])
snlsmures = np.copy(datamures[snls])

lowzmass = np.copy(datahostmass[lowz])
lowzc = np.copy(datac[lowz])
lowzx1 = np.copy(datax1[lowz])
lowzmures = np.copy(datamures[lowz])

print(len(sdssmass), len(snlsmass), len(lowzmass))

print(len(sdssmass[np.asarray([x < 7 for x in sdssmass])]),
      len(snlsmass[np.asarray([x < 7 for x in snlsmass])]),
      len(lowzmass[np.asarray([x < 7 for x in lowzmass])]))

print(len(sdssmass[np.asarray([x > 10 for x in np.abs(sdssmures)])]),
      len(snlsmass[np.asarray([x > 10 for x in np.abs(snlsmures)])]),
      len(lowzmass[np.asarray([x > 10 for x in np.abs(lowzmures)])]))
